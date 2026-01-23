import os
import time
import pandas as pd
import networkx as nx
import argparse
import numpy as np
import scipy.io as sio
import scipy.sparse as sp
from scipy.sparse import csr_matrix
import tensorflow as tf
import matplotlib.pyplot as plt

plt.rcParams.update({
    # "text.usetex": True,        # use system LaTeX for all text
    "font.family": "serif",     # optional: serif font
    "text.latex.preamble": r"\usepackage{amsmath}"  # enable math environments
})
from dt_task import *
import copy
from backpressure import *
import warnings
warnings.filterwarnings('ignore')



parser = argparse.ArgumentParser()
parser.add_argument('--datapath', default='../data_poisson_10', type=str, help='input data path.')
parser.add_argument('--out', default='../output', type=str, help='output data path.')
parser.add_argument('--root', default='..', type=str, help='Root dir of project.')
parser.add_argument('--radius', default=0.0, type=float, help='Interference radius.')
parser.add_argument('--gtype', default='poisson', type=str, help='graph type or dataset dir')
parser.add_argument('--sizes', default='50', type=str, help='List of network sizes V')
parser.add_argument('--lb', default=33, type=float, help='Burst multiplier.')
parser.add_argument('--ls', default=1, type=float, help='Streaming multiplier.')
parser.add_argument('--pburst', default=0.0, type=float, help='Probability of having a flow being bursty.')
parser.add_argument('--T', default=1000, type=int, help='Number of time slots.')
parser.add_argument('--M', default=1, type=int, help='Maximum iterations of Luby MIS.')
parser.add_argument('--debug', default=False, action='store_true', help='Only set to True while debugging locally')
parser.add_argument('--function', default='proportional', type=str, help='pheromone routing function')
parser.add_argument('--exploration_rate', default=0.0, type=float, help='Exploration rate for virtual ants')
parser.add_argument('--decay', default=0.998, type=float, help='decay rate of the pheromones')
parser.add_argument('--unit', default=0.01, type=float, help='unit being added to the pheromones')
parser.add_argument('--not_going_back', default=False, action='store_true', help='Ants not going back on a link')
parser.add_argument('--ph_diff', default=True, action='store_true', help='Differential pheromone on each directin of a link')
parser.add_argument('--init', default=0.01, type=float, help='Initial value of the pheromones')
parser.add_argument('--rmax', type=float, default=1.5, help='flow rate max')
parser.add_argument('--rmin', type=float, default=0.5, help='flow rate min')
args = parser.parse_args()


'''
DT parameters
'''
# M = 1
# M = 3
M = args.M
if M == 1:
    num_rounds = 1
elif M >= 3:
    num_rounds = -1

K = 5
alpha_dt = 0.5
# alpha = 1.0      # or 0.5 if you want damped updates
L_int = 128  # grid size for the integral
eps = 1e-12

# === Contention Control: Config ===
# every time slot, check the past 50 slots's moving average.
W_WINDOW = 50         # sliding window length
q_low = 50.0           # low-queue threshold for "light load"
q_high = 800.0        # high-queue threshold for congestion protection
q_panic = 1000.0       # hard safety: force gate to 1 if exceeded

eta = 0.5              # multiplicative decrease strength under overscheduling
alpha_cc = 0.8            # additive increase when high load (recover g -> 1)
rho = 0.01             # slow relaxation toward 1
g_min = 0.05           # min gate floor
tau_x = 0.10           # noise guard: ignore tiny overschedule
kappa_q = 20.0         # smoothing for low-load sigmoid
kappa_h = 100.0        # smoothing for high-load sigmoid
w_x = 1.0              # overscheduling weight
_eps_cc = 1e-6         # numerical epsilon for division
use_neighbor_smoothing = False
zeta = 0.0             # set to 0.1 if enabling neighbor smoothing
# Contention control margin: only trigger control when empirical duty > (1+margin)*predicted
CONTROL_MARGIN = 0.1  # 5% relative margin
# Enable debug prints for first few steps to inspect hat_x, x_DT, delta, S
CONTROL_DEBUG = False

'''
SP-only scheme implementation for DTscheduler
'''

debug = args.debug
std = 0.01
T = args.T
cf_radius = args.radius
datapath = args.datapath
ph_diff = args.ph_diff
val_mat_names = sorted(os.listdir(datapath))

sizes = [int(i) for i in args.sizes.split(',')]
sizes = sorted(sizes)
sizes_txt = [str(i) for i in sizes]
sizes_txt = '-'.join(sizes_txt)

load_bursty = args.lb
load_streaming = args.ls
pburst = args.pburst
print("pburst", pburst)

link_rate_max = 42  # 42
link_rate_min = 10  # 10
link_rate_avg = (link_rate_max + link_rate_min) / 2
arrival_max = args.rmax  
arrival_min = args.rmin  
arrival_avg = (arrival_min + arrival_max) / 2
if arrival_max == arrival_avg:
    postfix = "v31_thpt_{:.1f}".format(arrival_avg*load_streaming)
else:
    postfix = "v31_ls_{:.1f}".format(load_streaming)
burst_cutoff = 30


if pburst == 0.:
    bursty_info = 'streaming_s{}'.format(load_streaming)
else:
    bursty_info = 'mixed_pb{}_s{}_b{}'.format(pburst, load_streaming, load_bursty)

output_dir = args.out
# Output CSV file for SP-only
output_csv = os.path.join(output_dir, "sp_only_test_{}_T_{}_ir_{:.1f}_sizes_{}_link-{}_M_{}_{}.csv".format(
    datapath.split("/")[-1], T, cf_radius, sizes_txt, link_rate_avg, M, postfix))

if os.path.isfile(output_csv):
    df_res = pd.read_csv(output_csv, index_col=False)
else:
    df_res = pd.DataFrame(
        columns=[
            'filename', 'seed', 'num_nodes', 'm', 'T', 'cf_radius', 'cf_degree', 'f_case', 'num_flows', 'ls',
            'src', 'dst', 'flow_rate', "cutoff", "start",
            'opt', 'Algo',
            'src_delay_raw', 'est_delay_raw', 'delivery_raw', 'active_links', 'cnt_out_raw', 'cnt_in_raw',
            'runtime_sim', 'runtime_dt', 'dt_K'
        ]
    )

# Add separate CSV for joint probabilities (E x E matrix)
joint_prob_csv = os.path.join(output_dir, "joint_probabilities_{}_T_{}_ir_{:.1f}_sizes_{}_link-{}_{}.csv".format(
    datapath.split("/")[-1], T, cf_radius, sizes_txt, link_rate_avg, postfix))

if os.path.isfile(joint_prob_csv):
    df_joint = pd.read_csv(joint_prob_csv, index_col=False)
else:
    df_joint = pd.DataFrame(
        columns=[
            'filename', 'seed', 'num_nodes', 'm', 'T', 'cf_radius', 'f_case', 'num_flows',
            'link_i', 'link_j', 'src_i', 'dst_i', 'src_j', 'dst_j', 
            'joint_prob', 'marginal_prob_i', 'marginal_prob_j',
            'are_neighbors',     # whether links i and j conflict
            'opt', 'Algo', 'physical'
        ]
    )


def plot_duty_scatter(
    pred_duty,            # 1D np.ndarray or tf.Tensor, shape [E]
    emp_duty,             # 1D np.ndarray or tf.Tensor, shape [E]
    lambda_e_vec,         # 1D np.ndarray or tf.Tensor, shape [E]  (used to filter active links)
    title="Predicted vs Empirical Duty (active links only)",
    out_path=None,        # e.g., "/tmp/duty_scatter.png"; if None -> plt.show()
    show=False            # True to show() (useful locally); on servers prefer saving
):
    """Scatter plot of predicted (x) vs empirical (y) duty cycles for links with lambda > 0, with x=y dashed line."""
    # Convert to numpy
    to_np = lambda x: x.numpy() if tf.is_tensor(x) else np.asarray(x)
    x = to_np(pred_duty).astype(np.float32).reshape(-1)
    y = to_np(emp_duty).astype(np.float32).reshape(-1)
    lam = to_np(lambda_e_vec).astype(np.float32).reshape(-1)

    # Filter: only links with non-zero traffic and finite values
    mask = (lam > 0.0) & np.isfinite(x) & np.isfinite(y)
    x = x[mask]; y = y[mask]

    if x.size == 0:
        print("[plot_duty_scatter] No active links to plot (lambda_e_vec > 0). Skipping.")
        return

    # Axes limits & diag
    lo = float(min(x.min(), y.min(), 0.0))
    hi = float(max(x.max(), y.max(), 1.0))
    pad = 0.02 * (hi - lo + 1e-9)
    xmin, xmax = lo - pad, hi + pad
    ymin, ymax = xmin, xmax  # keep square

    # Plot
    fig = plt.figure(figsize=(5, 5))
    ax = fig.add_subplot(111)
    ax.scatter(x, y, s=25, alpha=0.6, c='red')
    ax.plot([xmin, xmax], [ymin, ymax], linestyle="--")  # x=y
    ax.set_xlim(xmin, xmax); ax.set_ylim(ymin, ymax)
    ax.set_xlabel("Predicted duty cycle")
    ax.set_ylabel("Empirical duty cycle")
    ax.set_title(f"{title}\nN={x.size}")

    # Optionally add simple metrics
    try:
        # Pearson R
        xv = x - x.mean(); yv = y - y.mean()
        r = float((xv @ yv) / (np.sqrt((xv**2).sum()) * np.sqrt((yv**2).sum()) + 1e-12))
        ax.text(0.02, 0.98, f"Pearson r={r:.3f}", transform=ax.transAxes, va="top")
    except Exception:
        pass

    fig.tight_layout()
    if out_path:
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        print(f"[plot_duty_scatter] Saved: {out_path}")
    elif show:
        plt.show()
    else:
        # Default to saving to a temporary name if neither provided
        tmp_path = "duty_scatter.png"
        fig.savefig(tmp_path, dpi=150)
        plt.close(fig)
        print(f"[plot_duty_scatter] Saved: {tmp_path}")


def plot_congestion_vs_queue(
    congestion_pred,        # 1D array or tensor of predicted congestion metric (non-negative)
    queue_last,             # 1D array or tensor of queue length at last time step (non-negative)
    lambda_e_vec=None,      # optional mask/filter (e.g. active links)
    link_fed_ratios=None,   # optional 1D np.ndarray, shape [E]
    title="Predicted Congestion vs Queue Length (last step)",
    out_path=None,          # path to save PNG
    show=False
):
    """
    Scatter plot: congestion prediction (x) vs queue length (y), both non-negative.
    Filters to active links (lambda_e_vec > 0) if provided.
    """

    # Convert to numpy
    to_np = lambda x: x.numpy() if tf.is_tensor(x) else np.asarray(x)
    x = to_np(congestion_pred).astype(np.float32).reshape(-1)
    y = to_np(queue_last).astype(np.float32).reshape(-1)

    # Optional filter for active links
    if lambda_e_vec is not None:
        lam = to_np(lambda_e_vec).astype(np.float32).reshape(-1)
        mask = lam > 0.0
        x = x[mask]; y = y[mask]

    # Sanity filter for finite and non-negative values
    mask = (x >= 0.0) & (y >= 0.0) & np.isfinite(x) & np.isfinite(y)
    x = x[mask]; y = y[mask]

    if x.size == 0:
        print("[plot_congestion_vs_queue] No valid data to plot.")
        return

    # Axis limits
    xpad = 0.05 * (x.max() - x.min() + 1e-9)
    ypad = 0.05 * (y.max() - y.min() + 1e-9)
    xmin = 0
    xmax = max(x.max() + xpad, 1.0)  # ensure upper bound ≥ 1.0
    ymin, ymax = max(0, y.min() - ypad), y.max() + ypad

    # Plot
    fig = plt.figure(figsize=(5, 4))
    ax = fig.add_subplot(111)

    # Optional color-coding by link_fed_ratios
    if link_fed_ratios is not None:
        c = np.asarray(link_fed_ratios).astype(np.float32).reshape(-1)
        # Clip to [0,1] and plot with colormap
        c = np.clip(c, 0.0, 1.0)
        sc = ax.scatter(x, y, c=c, cmap="jet_r", vmin=0, vmax=1.0, s=25, alpha=0.7)
        plt.colorbar(sc, ax=ax, label="Link feed ratio")
    else:
        ax.scatter(x, y, s=25, alpha=0.6)
    # ax.scatter(x, y, s=10, alpha=0.6)
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_xlabel(r"Predicted overload index: $\frac{\lambda_e}{r_e \, x_e}$")
    ax.set_ylabel("Queue length (last step)")
    ax.set_yscale("log")
    ax.set_ylim(1, 1e4)
    ax.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.7)
    ax.axvline(x=1.0, color="red", linestyle="--", linewidth=2.0)
    ax.set_title(f"{title}\nN={x.size}")
    fig.tight_layout()

    if out_path:
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        print(f"[plot_congestion_vs_queue] Saved: {out_path}")
    elif show:
        plt.show()
    else:
        tmp_path = "congestion_vs_queue.png"
        fig.savefig(tmp_path, dpi=150)
        plt.close(fig)
        print(f"[plot_congestion_vs_queue] Saved: {tmp_path}")


def init_pheromone_from_flows(bp_env, weight='delay'):
    ph = np.zeros_like(bp_env.pheromones)   # shape: (num_di_links, num_flows)
    # create other feature matrices
    lambda_e_vec = np.zeros((bp_env.num_links,), dtype=float)
    lambda_mtx = csr_matrix(bp_env.adj_i.shape, dtype=float)
    lambda_mtx_dok = lambda_mtx.todok()
    for fidx, flow in enumerate(bp_env.flows):
        src = flow.source_node
        dst = flow.dest_node
        fr = flow.arrival_rate
        try:
            path = nx.shortest_path(bp_env.graph_c, source=src, target=dst, weight=weight)
        except nx.NetworkXNoPath:
            continue
        if len(path) < 2:
            continue
        i_prev = -1 # previous link index
        fcol = fidx  # Each flow's column is its fidx
        for u, v in zip(path[:-1], path[1:]):
            if u >= bp_env.num_nodes or v >= bp_env.num_nodes:
                continue
            try:
                i = bp_env.link_list.index((u, v))
                di = i
            except ValueError:
                i = bp_env.link_list.index((v, u))
                di = i + bp_env.num_links
            ph[di, fcol] = 1.0
            # consecutive links
            if i_prev != -1:
                lambda_mtx_dok[i_prev, i] = lambda_mtx_dok.get((i_prev, i), 0.0) + fr
            i_prev = i
            lambda_e_vec[i] += fr
    lambda_mtx = lambda_mtx_dok.tocsr()
    bp_env.pheromones = ph
    return lambda_mtx, lambda_e_vec


# === for scheduling weight policy optimization ===

def z_from_u(u):
    z_raw = tf.exp(u)                           # > 0
    return z_raw / tf.reduce_mean(z_raw)        # mean(z) = 1

def edge_smoothness(u, src_idx, dst_idx):
    diff = tf.gather(u, tf.cast(src_idx, tf.int32)) - tf.gather(u, tf.cast(dst_idx, tf.int32))
    return tf.reduce_mean(tf.square(diff))


def causal_moving_average_at_t(c, t, W):
    """
    c : cumsum of x, shape (E, T)
    t : current time index (0 <= t < T)
    W : window size

    returns: vector of shape (E,)
             mean over x[:, max(0, t-W+1) : t+1]
    """
    # window bounds
    start = max(0, t - W + 1)
    end = t  # inclusive
    window_sum = c[:, start:end].sum(axis=1)
    window_len = max(end - start, 1)  # <= W near the beginning
    return window_sum / window_len


@tf.function
def dt_overload_from_u(u):
    """Run K iterations of your DT with z=exp(u)/mean(exp(u)); return final overload vector."""
    z_tf = z_from_u(u)   # [E]
    x_tf = x0_tf
    b_init_tf = tf.clip_by_value(lambda_e_tf / tf.maximum(x_tf * link_rates_tf, eps_tf), 0.0, 1.0)

    for _ in tf.range(K):
        if M==1:
            duty_tf, _ = analytical_duty_cycle_round1_tf_adj_edges(
                b_init=b_init_tf,
                src_idx=src_idx,
                dst_idx=dst_idx,
                z=z_tf,                  # <- priorities enter here
                b_ie_vals=None,          # independence case for now
                L=128,
                eps=1e-12,
                return_pwin=True,
            )
        else: 
            # multi-round
            duty_tf = analytical_duty_cycle_Mrounds_tf_adj_edges(
                b_init=b_init_tf,                           # 当前轮的entry概率
                src_idx=src_idx,                            # [K]
                dst_idx=dst_idx,                            # [K]
                z=z_tf,                                     # 优先级上界
                b_ie_vals_round1=None,       # 第一轮条件概率（固定）
                M=M,                                        # 调度轮数 (5轮)
                L=128,                                      # 积分网格
                eps=1e-12,
                return_pwin=False                           # 只需要duty cycle
            )
        # Update duty and marginals
        x_tf = (1.0 - alpha_dt) * x_tf + alpha_dt * duty_tf
        mu_tf = x_tf * link_rates_tf
        b_init_tf = tf.clip_by_value(lambda_e_tf / tf.maximum(mu_tf, eps_tf), 0.0, 1.0)

    overload_tf = tf.clip_by_value(lambda_e_tf / tf.maximum(mu_tf, eps_tf), 0.0, 1e6)
    return overload_tf, x_tf, z_tf  # return for logging/plotting

@tf.function
def loss_from_u(u):
    overload_tf, _, _ = dt_overload_from_u(u)
    # Smooth hinge on (overload-1)
    pos = overload_tf - 0.8
    L_over = tf.reduce_mean(tf.nn.softplus(kappa * pos) / kappa)

    # L2 on u (not on z) keeps priorities close to neutral
    L_l2 = lambda_l2 * tf.reduce_mean(tf.square(u))

    # Edge smoothness (optional)
    if lambda_smooth > 0.0:
        L_sm = lambda_smooth * edge_smoothness(u, src_idx, dst_idx)
    else:
        L_sm = 0.0

    return L_over + L_l2 + L_sm, {"L_over": L_over, "L_l2": L_l2, "L_sm": L_sm}

for id in range(len(val_mat_names)):
    filepath = os.path.join(datapath, val_mat_names[id])
    mat_contents = sio.loadmat(filepath)
    net_cfg = mat_contents['network'][0, 0]
    # link_rates = mat_contents["link_rate"][0]
    flows_cfg = mat_contents["flows"][0]
    pos_c = mat_contents["pos_c"]
    seed = int(net_cfg['seed'].flatten()[0])
    NUM_NODES = int(net_cfg['num_nodes'].flatten()[0])
    m = net_cfg['m'].flatten()[0]

    if NUM_NODES not in sizes:
        continue

    # Configuration
    if args.gtype.lower() == 'poisson':
        bp_env = BackpressureAnt(NUM_NODES, T, seed, m, pos_c, cf_radius=cf_radius, gtype=filepath)
    else:
        bp_env = BackpressureAnt(NUM_NODES, T, seed, m, pos_c, cf_radius=cf_radius, gtype=args.gtype)
    if not bp_env.connected:
        print("Unconnected {}".format(val_mat_names[id]))
        continue
    
    algo = 'DT-Opt-Ctrl'

    for f_case in range(10):
        # skip test if test results of current case already exist
        if not df_res.query(
                "@val_mat_names[{}] == filename and \
                @seed == seed and \
                @m == m and \
                @cf_radius == cf_radius and \
                @f_case == f_case and \
                @algo == Algo and \
                @NUM_NODES == num_nodes".format(id)
        ).empty:
            print("skip test case: {}, {}".format(val_mat_names[id], f_case))
            continue

        np.random.seed(seed * 10 + f_case)
        # flows_perc = np.random.randint(30, 50)
        flows_perc = np.random.randint(15, 25)
        num_flows = round(flows_perc / 100 * bp_env.num_nodes)
        nodes = bp_env.graph_c.nodes()
        num_arr = np.random.permutation(nodes)

        arrival_rates = np.random.uniform(arrival_min, arrival_max, (num_flows,))
        link_rates = np.random.uniform(link_rate_min, link_rate_max, size=(bp_env.num_links,))

        max_link_rate = np.amax(link_rates)
        delay_est = np.divide(link_rate_avg * max_link_rate, link_rates)
        for (src, dst), delay in zip(bp_env.link_list, delay_est):
            bp_env.graph_c[src][dst]["delay"] = delay
        cali_const = max_link_rate

        srcs = []
        dsts = []
        cutoffs = []
        flow_rates = []
        start_ts = []
        flows = []

        fll = 0
        for fidx in range(num_flows):
            src = num_arr[2 * fidx]
            dst = num_arr[2 * fidx + 1]
            cutoff = -1
            ar_multiplier = load_streaming
            # added for the icassp submission
            start_t = 0
            if np.random.uniform(0, 1) < pburst:
                fll += 1
                cutoff = burst_cutoff
                # added for the icassp submission
                start_t = np.random.randint(0, bp_env.T - 100 - cutoff)
                ar_multiplier = load_bursty
            flow = {'src': src, 'dst': dst, 'rate': arrival_rates[fidx]* ar_multiplier, 'start': start_t, 'cut': cutoff}
            flows.append(flow)
            srcs.append(src)
            dsts.append(dst)
            flow_rates.append(arrival_rates[fidx] * ar_multiplier)
            cutoffs.append(cutoff)
            start_ts.append(start_t)
        print(f'out of {num_flows} flows {fll} of them are bursty')

        # SP-only scheme (scheme 7)
        opt = 0  # Only physical routing

        # Physical routing only (vi = 1)
        vi = 1
        cT = T
        flows_vi = copy.deepcopy(flows)
        np.random.seed(seed * 10 + f_case)

        # Ant configuration
        func = args.function
        exploration_rate = args.exploration_rate
        decay = args.decay
        unit = args.unit
        not_going_back = args.not_going_back
        ph_diff = args.ph_diff
        init = args.init

        bp_env.t_recordings = [cT - 1]
        # configure flows and realize random instances
        bp_env.clear_all_flows()
        
        for fidx in range(num_flows):
            flow = flows_vi[fidx]
            bp_env.add_flow(flow['src'], flow['dst'], rate=arrival_rates[fidx] * ar_multiplier, start=flow['start'], cutoff=flow['cut'])

        bp_env.flows_init()
        bp_env.flows_reset()
        bp_env.links_init(link_rates)
        bp_env.queues_init()
        bp_env.pheromone_init(decay=decay, unit=unit, init=init)

        # Initialize pheromones with shortest paths (for opt=0)
        lambda_mtx, lambda_e_vec = init_pheromone_from_flows(bp_env, weight='delay')
        bp_env.freeze_pherom()  # Fixed strategy
        # active_e_vec = (lambda_e_vec > 0.0).astype(np.int64)
        active_e_vec = np.ones_like(lambda_e_vec)
        deg_eff = (bp_env.adj_i @ active_e_vec)  # dense vector
        deg_eff = np.asarray(deg_eff, dtype=float).reshape(-1)
        x_init = np.divide(active_e_vec.astype(float), active_e_vec.astype(float) + deg_eff)
        x_init = np.nan_to_num(x_init, nan=0.0)

        delay_mtx = np.zeros_like(bp_env.queue_matrix)

        shortest_paths = all_pairs_shortest_paths(bp_env.graph_c, weight='delay')
        bias_matrix = shortest_paths
        bias_vector = bp_env.bias_diff(bias_matrix)
        link_bias_vec = bias_vector * (link_rate_avg / np.min(delay_est))
        link_bias = None  # SP-only doesn't use bias

        # routing simulation
        active_links = np.zeros((T,))
        # Initialize 5-round scheduling statistics: link scheduling counts for each round
        round_sched_count = [np.zeros(bp_env.num_links, dtype=int) for _ in range(5)]
        link_contending = np.zeros((T, bp_env.num_links), dtype=bool)  # Shape: (T, E)

        # ==========================TF2 iterative analytical DT==========================
        # --- Hyperparameters (tune lightly) ---
        kappa = 5.0  # smooth hinge sharpness
        lambda_l2 = 1e-3  # L2 on u
        lambda_smooth = 1e-3  # edge smoothness (set 0 to disable)
        lr = 1e-2  # optimizer learning rate
        steps = 20  # descent steps

        # Build edge lists once
        src_idx_np, dst_idx_np, E = csr_to_edge_lists(bp_env.adj_i)
        trans_map = build_transpose_map(src_idx_np, dst_idx_np, E)
        src_idx = tf.constant(src_idx_np, tf.int32)
        dst_idx = tf.constant(dst_idx_np, tf.int32)

        # Non-trainable tensors
        x0_tf = tf.convert_to_tensor(x_init, tf.float32)  # initial duty
        link_rates_tf = tf.convert_to_tensor(link_rates, tf.float32)
        lambda_e_tf = tf.convert_to_tensor(lambda_e_vec, tf.float32)
        eps_tf = tf.constant(1e-12, tf.float32)

        # Trainable variable in unconstrained space u (so z = exp(u) > 0)
        # Initialize u such that z starts at 1.0
        u_tf = tf.Variable(tf.zeros([E], dtype=tf.float32), name="u_priorities")

        # ===dummy dt run===
        # --- Prepare DT inputs ---
        z_tf = z_from_u(u_tf)
        x_tf = x0_tf
        b_init_tf = tf.clip_by_value(lambda_e_tf / tf.maximum(x_tf * link_rates_tf, eps_tf), 0.0, 1.0)

        start_time = time.time()

        duty_tf = analytical_duty_cycle_round1_tf_adj_edges(
            b_init=b_init_tf,
            src_idx=src_idx,
            dst_idx=dst_idx,
            z=z_tf,
            b_ie_vals=None,
            # M=1,  # M=1 调度轮数
            L=128,
            eps=1e-12,
            return_pwin=False
        )
        runtime_dt = time.time() - start_time
        print(f"Runtime for single DT call (analytical_duty_cycle_Mrounds_tf_adj_edges): {runtime_dt} s")
        # ========================== Optimize scheduling weights =========================
        print("Running optimization (not timed)...")
        # Trainable variable in unconstrained space u (so z = exp(u) > 0)
        # Initialize u such that z starts at 1.0
        u_tf.assign(tf.zeros([E], dtype=tf.float32))
        # --- Optimize u (and thus z) ---
        optimizer = tf.keras.optimizers.Adam(lr)

        # Warmup call to compile graphs
        _ = dt_overload_from_u(u_tf)
        _ = loss_from_u(u_tf)

        for step in range(steps):
            with tf.GradientTape() as tape:
                loss, parts = loss_from_u(u_tf)
            grads = tape.gradient(loss, [u_tf])
            optimizer.apply_gradients(zip(grads, [u_tf]))

            if step % 10 == 0 or step == steps - 1:
                overload_now, x_now, z_now = dt_overload_from_u(u_tf)
                print(f"[step {step:03d}] loss={float(loss):.6f} "
                      f"L_over={float(parts['L_over']):.6f} "
                      f"L2={float(parts['L_l2']):.6f} "
                      f"Sm={float(parts['L_sm']):.6f} "
                      f"overload max/mean={float(tf.reduce_max(overload_now)):.3f}/"
                      f"{float(tf.reduce_mean(overload_now)):.3f}")

        z_tf = z_from_u(u_tf)
        x_tf = tf.convert_to_tensor(x_init, dtype=tf.float32)
        b_init_np, mu_np = compute_marginals_b(x_init, link_rates, lambda_e_vec, eps=eps, clip=True)  # b:[E], mu:[E]
        b_init_tf = tf.constant(b_init_np.astype(np.float32))
        for k in range(1, K+1):
            if M == 1:
                # Independence case (skip joints)
                duty_tf, pwin_tf = analytical_duty_cycle_round1_tf_adj_edges(
                    b_init=b_init_tf,
                    src_idx=src_idx,
                    dst_idx=dst_idx,
                    z=z_tf,
                    # b_ie_vals=tf.constant(b_ie_vals_np, tf.float32),  # uses b_i gathered from b_init
                    b_ie_vals=None,
                    L=L_int,
                    eps=1e-12,
                    return_pwin=True,
                )
            else:
                # multi-round
                duty_tf = analytical_duty_cycle_Mrounds_tf_adj_edges(
                    b_init=b_init_tf,                         
                    src_idx=src_idx,                            
                    dst_idx=dst_idx,                            
                    z=z_tf,                                     
                    b_ie_vals_round1=None,       
                    M=M,                                        
                    L=128,                                     
                    eps=1e-12,
                    return_pwin=False                        
                )

            x_tf = (1.0 - alpha_dt) * x_tf + alpha_dt * duty_tf
            mu_tf = x_tf * link_rates_tf
            overload_tf = lambda_e_tf / tf.maximum(mu_tf, eps)
            b_init_tf = tf.clip_by_value(overload_tf, 0.0, 1.0)

        # ========================== Simulation ==========================
        z_np = z_tf.numpy()
        start_time = time.time()

        # === Contention Control: Predicted duty baseline (from DT) ===
        x_DT = x_tf.numpy().astype(np.float32)      # predicted duty per edge
        x_DT = np.clip(x_DT, 0.0, 1.0)

        # === Contention Control: State ===
        E = bp_env.num_links
        g_gate = np.ones(E, dtype=np.float32)                      # gate g_e in [g_min, 1]
        sched_hist = np.zeros((bp_env.num_links, T), dtype=np.uint8)       # rolling scheduled mask (0/1)
        # Q_hist = np.zeros((W_WINDOW, E), dtype=np.float32)         # rolling per-link queue

        for t in range(T):
            bp_env.pkt_arrival(t)

            # SP-only uses ph_routing with opt=0
            W_amp, W_sign = bp_env.ph_routing(
                t, func=func, exploration_rate=exploration_rate,
                not_going_back=not_going_back,
                link_bias=link_bias,
                ph_diff=ph_diff,
            )

            bp_env.W[:, t] = W_amp
            bp_env.WSign[:, t] = W_sign

            active_links[t] = np.count_nonzero(W_amp)
            utility = bp_env.W[:, t] * bp_env.link_rates[:, t]

            # Record contention for ALL links
            link_contending[t, :] = utility > 0
            ##############
            # === Contention Control: Update gate and probabilistically gate contention ===
            # 1) Compute windowed averages (use previous rows; current row will be filled after scheduling)
            hat_x = causal_moving_average_at_t(sched_hist.astype(np.float32), t, W_WINDOW) # empirical duty over last W_WINDOW

            # ----- Simplified per-time-step mask-based control -----
            # Block links whose moving-window mean empirical duty exceeds (1+CONTROL_MARGIN)*x_DT
            overshoot_mask = hat_x > (1.0 + CONTROL_MARGIN) * x_DT

            # Overshoot only when transmitter total backlog exceed average link rate
            # backlog_per_node = bp_env.queue_matrix.sum(axis=1)
            # tx_nodes = [u if s > 0 else v for (u, v), s in zip(bp_env.link_list, W_sign)]
            q_mask = W_amp < 1.1*link_rates
            overshoot_mask = overshoot_mask * q_mask

            utility[overshoot_mask] = 0.0

            ## 1-round and redraw scheduling
            mwis, round_solutions = bp_env.scheduling_rounds_redraw(utility, weights=z_np, n_rounds=num_rounds)
            # mwis, round_solutions = bp_env.scheduling_rounds_redraw(utility_gated, weights=z_np, n_rounds=num_rounds)

            bp_env.sched_count[mwis] += 1  # count how many times each undirected link is scheduled
            bp_env.transmission_ph(t, mwis)

            # === Contention Control: write current slot into windows ===
            sched_hist[mwis, t] += 1

        
        runtime_sim = time.time() - start_time
        # Calculate marginal probabilities for all links
        marginal_probs = link_contending.mean(axis=0).astype(np.float32)  # shape: (E,)

        # Vectorized joints only on adj nonzeros
        A_coo = bp_env.adj_i.tocoo(copy=False)
        rows = A_coo.row.astype(np.int32)  # e
        cols = A_coo.col.astype(np.int32)  # i

        # (optional) drop diagonal if your adj has no self-edges anyway
        mask_offdiag = rows != cols
        rows = rows[mask_offdiag]
        cols = cols[mask_offdiag]

        # Gather the two column-views at once: shapes (T, K)
        # For booleans, AND == multiplication after cast; we can use boolean AND directly.
        Xe = link_contending[:, rows]
        Xi = link_contending[:, cols]

        # Mean over time -> empirical joint probabilities per (e,i)
        joint_vals = (Xe & Xi).mean(axis=0).astype(np.float32)  # shape: (K,)

        # Build a sparse COO with exactly those entries
        joint_probs_sparse = sp.coo_matrix((joint_vals, (rows, cols)), shape=(bp_env.num_links, bp_env.num_links))
        joint_probs = joint_probs_sparse.todense()

        print(f"\n=== Network {val_mat_names[id]}, Case {f_case} ===")
        print(f"Links: {bp_env.num_links}")
        print(f"Marginal probabilities range: [{np.min(marginal_probs):.4f}, {np.max(marginal_probs):.4f}]")
        print(f"Joint probabilities range: [{np.min(joint_vals):.4f}, {np.max(joint_vals):.4f}]")

        # Collect realized link traffic
        link_thpt_emp = bp_env.link_comd_cnts.sum(axis=1).astype(float) / float(T)
        
        # Collect performance metrics
        cnt_in, cnt_out, delay_e2e, delay_e2e_raw, jitter_e2e, undeliver, delay_est = bp_env.collect_delay(opt, cT)
        duty_emp = bp_env.sched_count / T

        src_delay_mean = np.nanmean(delay_e2e)
        src_delay_max = np.nanmax(delay_e2e)
        src_delay_std = np.nanstd(delay_e2e)
        src_jitter_mean = np.nanmean(jitter_e2e)
        src_jitter_max = np.nanmax(jitter_e2e)
        src_jitter_std = np.nanstd(jitter_e2e)
        delivery_raw = np.divide(cnt_out.astype(float), cnt_in.astype(float))
        delivery_mean = np.nanmean(delivery_raw)
        delivery_max = np.nanmax(delivery_raw)
        delivery_std = np.nanstd(delivery_raw)

        print("{}: n {}, f {}, s {}, cf_deg {:.3f}, c {}, ".format(val_mat_names[id], NUM_NODES, num_flows,
                                                                    seed, bp_env.mean_conflict_degree,
                                                                    f_case),
                "opt {}, runtime {:.2f}, links {:.1f}".format(opt, runtime_sim, np.nanmean(active_links)),
                "Delay: mean {:.3f}, max {:.3f}, std {:.3f}".format(src_delay_mean, src_delay_max,
                                                                    src_delay_std),
                "Jitter: mean {:.3f}, max {:.3f}, std {:.3f}".format(src_jitter_mean, src_jitter_max,
                                                                    src_jitter_std),
                "Delivery: mean {:.3f}, max {:.3f}, std {:.3f}".format(delivery_mean, delivery_max,
                                                                        delivery_std),
                )

        try:
            # lambda_e_vec should be available from init_pheromone_from_flows
            lam = np.asarray(lambda_e_vec).reshape(-1)
            duty_np = np.asarray(duty_emp).reshape(-1)
            mask = (lam > 0.0) & np.isfinite(duty_np)
            duty_clean = duty_np[mask]

            if duty_clean.size == 0:
                d_count = 0
                d_min = np.nan
                d_q10 = np.nan
                d_q25 = np.nan
                d_median = np.nan
                d_q75 = np.nan
                d_q90 = np.nan
                d_max = np.nan
                d_mean = np.nan
                d_std = np.nan
            else:
                d_count = int(duty_clean.size)
                d_min = float(np.nanmin(duty_clean))
                d_q10 = float(np.nanpercentile(duty_clean, 10.0))
                d_q25 = float(np.nanpercentile(duty_clean, 25.0))
                d_median = float(np.nanpercentile(duty_clean, 50.0))
                d_q75 = float(np.nanpercentile(duty_clean, 75.0))
                d_q90 = float(np.nanpercentile(duty_clean, 90.0))
                d_max = float(np.nanmax(duty_clean))
                d_mean = float(np.nanmean(duty_clean))
                d_std = float(np.nanstd(duty_clean))

            duty_row = {
                'filename': val_mat_names[id],
                'seed': seed,
                'num_nodes': NUM_NODES,
                'm': m,
                'f_case': f_case,
                'Algo': algo,
                'runtime_sim': runtime_sim,
                'duty_count': d_count,
                'duty_min': d_min,
                'duty_q10': d_q10,
                'duty_q25': d_q25,
                'duty_median': d_median,
                'duty_q75': d_q75,
                'duty_q90': d_q90,
                'duty_max': d_max,
                'duty_mean': d_mean,
                'duty_std': d_std,
            }

            duty_inst_csv = os.path.join(output_dir, f"instance_duty_stats_{datapath.split('/')[-1]}_size{NUM_NODES}_T{T}_ir{cf_radius}_M{M}_{postfix}.csv")
            df_duty_inst = pd.DataFrame([duty_row])
            write_header_d = not os.path.isfile(duty_inst_csv)
            df_duty_inst.to_csv(duty_inst_csv, mode='a', header=write_header_d, index=False)
            print(f"[instance duty CSV] Appended duty summary to: {duty_inst_csv}")
        except Exception as ex:
            print(f"[instance duty CSV] Failed to write duty CSV for {val_mat_names[id]}, case {f_case}: {ex}")

        # plot_duty_scatter(
        #     duty_tf,  # tf.Tensor [E]
        #     duty_emp,
        #     lambda_e_vec,
        #     title=f"Analytical DT (TF2) vs Empirical (Load: {load_streaming:.1f})",
        #     out_path=os.path.join(output_dir, f"duty_scatter_tf_case{f_case:03d}_{load_streaming:.1f}_sch_M{M}_num{num_rounds}.png"),
        # )

        link_fed_ratios = np.divide(link_thpt_emp + W_amp/T, lambda_e_vec)
        link_fed_ratios_copy = link_fed_ratios.copy()
        link_fed_ratios = link_fed_ratios[lambda_e_vec > 0]

        # plot_congestion_vs_queue(
        #     overload_tf,
        #     W_amp,
        #     lambda_e_vec=lambda_e_vec,
        #     link_fed_ratios=link_fed_ratios,
        #     title=f"Congestion prediction vs Queue Length (Load: {load_streaming:.1f})",
        #     out_path=os.path.join(output_dir, f"congestion_vs_queue_case{f_case:03d}_{load_streaming:.1f}_sch_M{M}_num{num_rounds}.png"),
        # )

        try:
            # Try to get queue data from environment; accept any 2D-like structure
            q_raw = W_amp
            q_arr = None
            # if q_raw is None:
            #     # fallback: try other possible attributes
            #     q_raw = getattr(bp_env, 'queues', None)
            if q_raw is not None:
                q_arr = np.asarray(q_raw)

            # compute flattened, finite values
            if q_arr is None or q_arr.size == 0:
                q_flat = np.array([])
            else:
                q_flat = q_arr.reshape(-1)
            q_clean = q_flat[np.isfinite(q_flat)]

            if q_clean.size == 0:
                q_count = 0
                q_min = np.nan
                q_q10 = np.nan
                q_q25 = np.nan
                q_median = np.nan
                q_q75 = np.nan
                q_q90 = np.nan
                q_max = np.nan
                q_mean = np.nan
                q_std = np.nan
            else:
                q_count = int(q_clean.size)
                q_min = float(np.nanmin(q_clean))
                q_q10 = float(np.nanpercentile(q_clean, 10.0))
                q_q25 = float(np.nanpercentile(q_clean, 25.0))
                q_median = float(np.nanpercentile(q_clean, 50.0))
                q_q75 = float(np.nanpercentile(q_clean, 75.0))
                q_q90 = float(np.nanpercentile(q_clean, 90.0))
                q_max = float(np.nanmax(q_clean))
                q_mean = float(np.nanmean(q_clean))
                q_std = float(np.nanstd(q_clean))

            inst_row = {
                'filename': val_mat_names[id],
                'seed': seed,
                'num_nodes': NUM_NODES,
                'm': m,
                'f_case': f_case,
                # Use explicit Algo label for these congestion summaries
                'Algo': algo,
                'runtime_sim': runtime_sim,
                'queue_count': q_count,
                'queue_min': q_min,
                'queue_q10': q_q10,
                'queue_q25': q_q25,
                'queue_median': q_median,
                'queue_q75': q_q75,
                'queue_q90': q_q90,
                'queue_max': q_max,
                'queue_mean': q_mean,
                'queue_std': q_std,
            }

            inst_csv = os.path.join(output_dir, f"instance_queue_stats_{datapath.split('/')[-1]}_size{NUM_NODES}_T{T}_ir{cf_radius}_M{M}_{postfix}.csv")
            df_inst = pd.DataFrame([inst_row])
            write_header = not os.path.isfile(inst_csv)
            df_inst.to_csv(inst_csv, mode='a', header=write_header, index=False)
            print(f"[instance CSV] Appended congestion summary to: {inst_csv}")
        except Exception as ex:
            print(f"[instance CSV] Failed to write congestion CSV for {val_mat_names[id]}, case {f_case}: {ex}")

        # ========================per-link recording=============================
        # Record per-link rows only for active links. Add fields: Algo, W_amp (mean over time),
        # duty_emp (empirical duty), link_fed_ratio, overload_tf (predicted overload).
        link_data = []
        
        for link_idx in range(E):
            src_node, dst_node = bp_env.link_list[link_idx]
            is_scheduled = lambda_e_vec[link_idx] > 0

            if not is_scheduled:
                continue
            
            to_np = lambda x: x.numpy() if tf.is_tensor(x) else np.asarray(x)
            duty_tf = to_np(duty_tf).astype(np.float32).reshape(-1)
            duty_emp = to_np(duty_emp).astype(np.float32).reshape(-1)
            lam = to_np(lambda_e_vec).astype(np.float32).reshape(-1)
            overload = to_np(overload_tf).astype(np.float32).reshape(-1)

            link_record = {
                'link_idx': link_idx,
                'src_node': src_node,
                'dst_node': dst_node,
                'overload_tf': overload[link_idx],
                'lambda_e_vec': lam[link_idx],
                'W_amp': W_amp[link_idx],
                'duty_emp': duty_emp[link_idx],
                'duty_tf': duty_tf[link_idx],
                'link_fed_ratio': link_fed_ratios_copy[link_idx],
            }

            link_data.append(link_record)
        output_dir_link = os.path.join(output_dir, os.path.basename(datapath))
        os.makedirs(output_dir_link, exist_ok=True)

        link_csv_case = os.path.join(
            output_dir_link,
            f"link_stats_{val_mat_names[id].replace('.mat','')}_M{M}_cf{cf_radius}_case{f_case}_ls{load_streaming}_algo{algo}.csv"
        )
        df_link_new = pd.DataFrame(link_data)
        df_link_new.to_csv(link_csv_case, index=False)
        print(f"[per-link CSV] Saved: {link_csv_case}")

        result = {
            "filename": val_mat_names[id],
            "seed": seed,
            "num_nodes": NUM_NODES,
            "m": m,
            "T": cT,
            "cf_radius": cf_radius,
            "cf_degree": bp_env.mean_conflict_degree,
            "opt": opt,
            "Algo": algo,
            "f_case": f_case,
            "num_flows": bp_env.num_flows,
            "ls": load_streaming,
            "src_delay_raw": delay_e2e,
            "est_delay_raw": delay_est,
            "delivery_raw": delivery_raw,
            "active_links": np.nanmean(active_links),
            "cnt_out_raw": cnt_out,
            "cnt_in_raw": cnt_in,
            "src": srcs,
            "dst": dsts,
            "flow_rate": flow_rates,
            "cutoff": cutoffs,
            "start": start_ts,
            "runtime_sim": runtime_sim,
            "runtime_dt": runtime_dt,
            "dt_K": K,
        }
        new_row = pd.DataFrame(result)
        df_res = pd.concat([df_res, new_row], ignore_index=True)
        df_res.to_csv(output_csv, index=False)

        # only run 1 case on one graph for debugging
        if debug:
            break
    
    if debug:
        break

print(f'Done')