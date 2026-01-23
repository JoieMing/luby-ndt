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
import glob



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
M = 3
# M = 3
K = 5
alpha = 0.5
# alpha = 1.0      # or 0.5 if you want damped updates
L_int = 128  # grid size for the integral
eps = 1e-12

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
    postfix = "_thpt_{:.1f}".format(arrival_avg*load_streaming)
else:
    postfix = "_ls_{:.1f}".format(load_streaming)
burst_cutoff = 30


if pburst == 0.:
    bursty_info = 'streaming_s{}'.format(load_streaming)
else:
    bursty_info = 'mixed_pb{}_s{}_b{}'.format(pburst, load_streaming, load_bursty)

output_dir = args.out
# Output CSV file for SP-only
output_csv = os.path.join(output_dir, "sp_only_test_{}_T_{}_ir_{:.1f}_sizes_{}_link-{}_{}.csv".format(
    datapath.split("/")[-1], T, cf_radius, sizes_txt, link_rate_avg, postfix))
# Create duty cycle CSV filename
duty_csv = os.path.join(output_dir, "duty_cycle_{}_T_{}_ir_{:.1f}_sizes_{}_link-{}_{}_dt_small.csv".format(
    datapath.split("/")[-1], T, cf_radius, sizes_txt, link_rate_avg, postfix))

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

# Initialize duty cycle dataframe if it doesn't exist
if os.path.isfile(duty_csv):
    df_duty = pd.read_csv(duty_csv, index_col=False)
else:
    df_duty = pd.DataFrame(
        columns=[
            'filename', 'seed', 'num_nodes', 'm',  'cf_radius', 'f_case', 'num_flows', 'ls',
            # 'alpha',
            'link_idx', 'src_node', 'dst_node', 
            'duty_cycle', 'est_duty_cycle', 
            'marginal_prob',  
            'is_scheduled',
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

@tf.function
def dt_overload_from_u(u):
    """Run K iterations of your DT with z=exp(u)/mean(exp(u)); return final overload vector."""
    z_tf = z_from_u(u)   # [E]
    x_tf = x0_tf
    b_init_tf = tf.clip_by_value(lambda_e_tf / tf.maximum(x_tf * link_rates_tf, eps_tf), 0.0, 1.0)

    for _ in tf.range(K):
        # duty_tf, _ = analytical_duty_cycle_round1_tf_adj_edges(
        #     b_init=b_init_tf,
        #     src_idx=src_idx,
        #     dst_idx=dst_idx,
        #     z=z_tf,                  # <- priorities enter here
        #     b_ie_vals=None,          # independence case for now
        #     L=128,
        #     eps=1e-12,
        #     return_pwin=True,
        # )
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
        x_tf = (1.0 - alpha) * x_tf + alpha * duty_tf
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

    # for f_case in range(1):
    # Before iterating f_case, collect any existing runtime_summary_*.csv
    # (search recursively under output_dir) so we can skip already-run cases.
    pattern_runtime = os.path.join(output_dir, '**', 'runtime_summary_*.csv')
    runtime_files = glob.glob(pattern_runtime, recursive=True)
    if runtime_files:
        try:
            existing_runtime = pd.concat([pd.read_csv(f, index_col=False) for f in runtime_files], ignore_index=True)
            # Normalize column names if necessary
            existing_runtime.columns = [c.strip() for c in existing_runtime.columns]
        except Exception as e:
            print(f"[warning] Failed reading existing runtime_summary files: {e}")
            existing_runtime = pd.DataFrame()
    else:
        existing_runtime = pd.DataFrame()

    for f_case in range(10):
        # skip test if test results of current case already exist
        if not df_duty.query(
                "@val_mat_names[{}] == filename and \
                @seed == seed and \
                @m == m and \
                @cf_radius == cf_radius and \
                @f_case == f_case and \
                @NUM_NODES == num_nodes".format(id)
        ).empty:
            print("skip test case: {}, {}".format(val_mat_names[id], f_case))
            continue

        # Also skip if a runtime_summary file already contains this (filename, seed, m, cf_radius, f_case, num_nodes)
        if not existing_runtime.empty:
            try:
                mask = (
                    (existing_runtime['filename'] == val_mat_names[id]) &
                    (existing_runtime['seed'] == seed) &
                    (existing_runtime['m'] == m) &
                    (existing_runtime['cf_radius'] == cf_radius) &
                    (existing_runtime['f_case'] == f_case) &
                    (existing_runtime['num_nodes'] == NUM_NODES)
                )
                if mask.any():
                    print(f"skip test case (found in runtime_summary): {val_mat_names[id]}, {f_case}")
                    continue
            except Exception:
                # If structure doesn't match, ignore and continue running (safer)
                pass

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
        algo = 'SP+Unweighted Luby'
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

        duty_tf = analytical_duty_cycle_Mrounds_tf_adj_edges(
            b_init=b_init_tf,
            src_idx=src_idx,
            dst_idx=dst_idx,
            z=z_tf,
            b_ie_vals_round1=None,
            M=5,  # M=1 调度轮数
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

        # 2) Inputs for round-1
        # b_init_tf = tf.constant(marginal_probs.astype(np.float32))  # [E]
        # Optional per-link scales
        # z_tf = None  # or tf.constant(z_np, tf.float32)
        # b_ie_vals_np = make_b_ie_from_joint(marginal_probs, joint_probs_sparse, src_idx_np, dst_idx_np, eps=1e-12)
        z_tf = z_from_u(u_tf)

        x_tf = tf.convert_to_tensor(x_init, dtype=tf.float32)
        b_init_np, mu_np = compute_marginals_b(x_init, link_rates, lambda_e_vec, eps=eps, clip=True)  # b:[E], mu:[E]
        b_init_tf = tf.constant(b_init_np.astype(np.float32))
        for k in range(1, K+1):
            # Independence case (skip joints)
            # duty_tf, pwin_tf = analytical_duty_cycle_round1_tf_adj_edges(
            #     b_init=b_init_tf,
            #     src_idx=src_idx,
            #     dst_idx=dst_idx,
            #     z=z_tf,
            #     # b_ie_vals=tf.constant(b_ie_vals_np, tf.float32),  # uses b_i gathered from b_init
            #     b_ie_vals=None,
            #     L=L_int,
            #     eps=1e-12,
            #     return_pwin=True,
            # )
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

            x_tf = (1.0 - alpha) * x_tf + alpha * duty_tf
            mu_tf = x_tf * link_rates_tf
            overload_tf = lambda_e_tf / tf.maximum(mu_tf, eps)
            b_init_tf = tf.clip_by_value(overload_tf, 0.0, 1.0)

        # ========================== Simulation ==========================
        z_np = z_tf.numpy()
        start_time = time.time()

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

            # num_rounds = 1
            num_rounds = 5
            active_links[t] = np.count_nonzero(W_amp)
            utility = bp_env.W[:, t] * bp_env.link_rates[:, t]

            # Record contention for ALL links
            link_contending[t, :] = utility > 0
            ##############
            ## 1-round and redraw scheduling
            mwis, round_solutions = bp_env.scheduling_rounds_redraw(utility, weights=z_np, n_rounds=num_rounds)

            bp_env.sched_count[mwis] += 1  # count how many times each undirected link is scheduled
            bp_env.transmission_ph(t, mwis)
        
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
        # -------------------------------
        # Luby's Digital Twin (Algorithm)
        # -------------------------------

        E = bp_env.num_links

        # priority weights
        z_vec = np.ones(E, dtype=float)
        # initiate the duty cycle: x^{(0)}_e = z_e / (z_e + sum_{i in N(e)} z_i)
        x = np.zeros(E, dtype=float)
        for e in range(E):
            nbrs = bp_env.adj_i[e].nonzero()[1]
            denom = z_vec[e] + z_vec[nbrs].sum()
            x[e] = z_vec[e] / max(denom, eps)
        x = np.clip(x, 0.0, 1.0)

        # # DT fixed-point iteration
        # for k_iter in range(1, K+1):
        #     # μ^{(k-1)} = r ⊙ x^{(k-1)}  (effective service rate)
        #     mu_eff = np.maximum(link_rates * x, eps)
        #     # b^{(k)} = min(λ / μ^{(k-1)}, 1) computed by reusing your helper:
        #     # pass effective capacities = μ^{(k-1)} as 'link_rates'
        #     b_k, lambda_k = bp_env.estimate_link_busylevel(weight='delay', link_rates=mu_eff)  # shape (E,)

        #     # update duty cycle x^{(k)} 
        #     # ==========================5-round test==========================
        #     # redraw
        #     x_star, P_win_round, b_round = bp_env.compute_analytical_duty_cycle_v3redraw(b_init=b_k, b_joint_1=b_joint_1, M=M, L=L_int)

        #     # relaxation: x^{(k)} -> x^{(k+1)}
        #     x = (1.0 - alpha) * x + alpha * x_star
        #     x = np.clip(x, 0.0, 1.0)
        
        # ==========================single-round DT test==========================
        x_star, P_win_round, b_round = bp_env.compute_analytical_duty_cycle_v3redraw(b_init=marginal_probs, b_joint_1=joint_probs, M=M, L=L_int)

        # relaxation: x^{(k)} -> x^{(k+1)}
        # x = (1.0 - alpha) * x + alpha * x_star
        x = x_star
        duty_est_np = np.clip(x, 0.0, 1.0)

        # plot_duty_scatter(
        #     duty_est_np,
        #     duty_emp,  # empirical from simulation (numpy)
        #     lambda_e_vec,  # to filter active links
        #     title=f"Analytical DT (NumPy) vs Empirical (Load: {load_streaming:.1f})",
        #     out_path=os.path.join(output_dir, f"duty_scatter_numpy_case{f_case:03d}_{load_streaming:.1f}_sch_M{M}_num{num_rounds}.png"),
        # )

        # plot_duty_scatter(
        #     duty_tf,  # tf.Tensor [E]
        #     duty_emp,
        #     lambda_e_vec,
        #     title=f"Analytical DT (TF2) vs Empirical (Load: {load_streaming:.1f})",
        #     out_path=os.path.join(output_dir, f"duty_scatter_tf_case{f_case:03d}_{load_streaming:.1f}_sch_M{M}_num{num_rounds}.png"),
        # )

        link_fed_ratios = np.divide(link_thpt_emp + W_amp/T, lambda_e_vec)
        link_fed_ratios = link_fed_ratios[lambda_e_vec > 0]

        # plot_congestion_vs_queue(
        #     overload_tf,
        #     W_amp,
        #     lambda_e_vec=lambda_e_vec,
        #     link_fed_ratios=link_fed_ratios,
        #     title=f"Congestion prediction vs Queue Length (Load: {load_streaming:.1f})",
        #     out_path=os.path.join(output_dir, f"congestion_vs_queue_case{f_case:03d}_{load_streaming:.1f}_sch_M{M}_num{num_rounds}.png"),
        # )

        pass

        # Save runtime summary for cross-load analysis
        runtime_summary_csv = os.path.join(output_dir, "runtime_summary_{}_T_{}_ir_{:.1f}_sizes_{}_link-{}_{}.csv".format(
            datapath.split("/")[-1], T, cf_radius, sizes_txt, link_rate_avg, postfix))
        
        if os.path.isfile(runtime_summary_csv):
            df_runtime_summary = pd.read_csv(runtime_summary_csv, index_col=False)
        else:
            df_runtime_summary = pd.DataFrame(columns=[
                'filename', 'seed', 'num_nodes', 'm', 'cf_radius', 'f_case', 'load_streaming',
                'runtime_dt', 'runtime_sim', 'speedup', 'num_flows'
            ])
        
        # Add current runtime data
        runtime_row = {
            'filename': val_mat_names[id],
            'seed': seed,
            'num_nodes': NUM_NODES,
            'm': m,
            'cf_radius': cf_radius,
            'f_case': f_case,
            'load_streaming': load_streaming,
            'runtime_dt': runtime_dt,
            'runtime_sim': runtime_sim,
            'speedup': runtime_sim / runtime_dt if runtime_dt > 0 else float('inf'),
            'num_flows': num_flows
        }
        
        df_runtime_summary = pd.concat([df_runtime_summary, pd.DataFrame([runtime_row])], ignore_index=True)
        df_runtime_summary.to_csv(runtime_summary_csv, index=False)

        # # only run 1 case on one graph for debugging
        # if debug:
        #     break
    
    # Generate runtime statistics and box plots for current network
    if len(df_res) > 0:
        # Get all f_case results for current network
        current_network_data = df_res[df_res['filename'] == val_mat_names[id]]
        
        if len(current_network_data) >= 2:  # Need at least 2 data points for statistics
            # Get unique runtime values per f_case (since each flow repeats the same runtime values)
            unique_data = current_network_data.drop_duplicates(subset=['f_case'])
            dt_runtimes = unique_data['runtime_dt'].values
            sim_runtimes = unique_data['runtime_sim'].values
            
            # Calculate statistics (quantiles)
            dt_stats = {
                'median': np.median(dt_runtimes),
                'q25': np.percentile(dt_runtimes, 25),
                'q75': np.percentile(dt_runtimes, 75),
                'q10': np.percentile(dt_runtimes, 10),
                'q90': np.percentile(dt_runtimes, 90),
                'mean': np.mean(dt_runtimes),
                'std': np.std(dt_runtimes),
                'min': np.min(dt_runtimes),
                'max': np.max(dt_runtimes)
            }
            
            sim_stats = {
                'median': np.median(sim_runtimes),
                'q25': np.percentile(sim_runtimes, 25),
                'q75': np.percentile(sim_runtimes, 75),
                'q10': np.percentile(sim_runtimes, 10),
                'q90': np.percentile(sim_runtimes, 90),
                'mean': np.mean(sim_runtimes),
                'std': np.std(sim_runtimes),
                'min': np.min(sim_runtimes),
                'max': np.max(sim_runtimes)
            }
            
            # Generate box plot
            fig, ax = plt.subplots(figsize=(10, 6))
            box_data = [dt_runtimes, sim_runtimes]
            labels = ['DT Optimization', 'Simulation']
            
            bp = ax.boxplot(box_data, labels=labels, patch_artist=True, 
                           showfliers=True, showmeans=True)
            
            # Customize colors
            bp['boxes'][0].set_facecolor('lightblue')
            bp['boxes'][1].set_facecolor('lightcoral')
            bp['means'][0].set_markerfacecolor('darkblue')
            bp['means'][1].set_markerfacecolor('darkred')
            
            ax.set_ylabel('Runtime (seconds)')
            ax.set_title(f'Runtime Distribution - {val_mat_names[id]}\n(n={len(dt_runtimes)} trials, Load: {load_streaming})')
            ax.grid(True, alpha=0.3)
            
            # Add quantile information text
            info_text = f"""DT Runtime Statistics:
            Median: {dt_stats['median']:.4f}s
            Q10-Q90: [{dt_stats['q10']:.4f}s, {dt_stats['q90']:.4f}s]
            Q25-Q75: [{dt_stats['q25']:.4f}s, {dt_stats['q75']:.4f}s]
            Mean ± Std: {dt_stats['mean']:.4f}s ± {dt_stats['std']:.4f}s

            Simulation Runtime Statistics:
            Median: {sim_stats['median']:.4f}s
            Q10-Q90: [{sim_stats['q10']:.4f}s, {sim_stats['q90']:.4f}s]
            Q25-Q75: [{sim_stats['q25']:.4f}s, {sim_stats['q75']:.4f}s]
            Mean ± Std: {sim_stats['mean']:.4f}s ± {sim_stats['std']:.4f}s

            Median Speedup: {sim_stats['median']/dt_stats['median']:.1f}x"""
            
            ax.text(0.02, 0.98, info_text, transform=ax.transAxes, 
                    verticalalignment='top', fontfamily='monospace', fontsize=9,
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.9))
            
            plt.tight_layout()
            
            # Save box plot
            box_plot_path = os.path.join(output_dir, f"runtime_boxplot_{val_mat_names[id]}_load{load_streaming}.png")
            fig.savefig(box_plot_path, dpi=150, bbox_inches='tight')
            plt.close(fig)
            
            # Print detailed statistics
            print(f"\n" + "="*60)
            print(f"RUNTIME STATISTICS: {val_mat_names[id]} (Load: {load_streaming})")
            print("="*60)
            print(f"Number of trials: {len(dt_runtimes)}")
            print(f"\nDT Optimization Runtime:")
            print(f"  Median:     {dt_stats['median']:.4f}s")
            print(f"  Mean:       {dt_stats['mean']:.4f}s ± {dt_stats['std']:.4f}s")
            print(f"  Q10-Q90:    [{dt_stats['q10']:.4f}s, {dt_stats['q90']:.4f}s]")
            print(f"  Q25-Q75:    [{dt_stats['q25']:.4f}s, {dt_stats['q75']:.4f}s]")
            print(f"  Min-Max:    [{dt_stats['min']:.4f}s, {dt_stats['max']:.4f}s]")
            
            print(f"\nSimulation Runtime:")
            print(f"  Median:     {sim_stats['median']:.4f}s")
            print(f"  Mean:       {sim_stats['mean']:.4f}s ± {sim_stats['std']:.4f}s")
            print(f"  Q10-Q90:    [{sim_stats['q10']:.4f}s, {sim_stats['q90']:.4f}s]")
            print(f"  Q25-Q75:    [{sim_stats['q25']:.4f}s, {sim_stats['q75']:.4f}s]")
            print(f"  Min-Max:    [{sim_stats['min']:.4f}s, {sim_stats['max']:.4f}s]")
            
            print(f"\nSpeedup Analysis:")
            speedups = sim_runtimes / dt_runtimes
            print(f"  Median Speedup: {np.median(speedups):.1f}x")
            print(f"  Mean Speedup:   {np.mean(speedups):.1f}x ± {np.std(speedups):.1f}x")
            print(f"  Speedup Range:  [{np.min(speedups):.1f}x, {np.max(speedups):.1f}x]")
            
            print(f"\nBox plot saved: {box_plot_path}")
            print("="*60)
    
    # if debug:
    #     break

# ========================== Cross-Load Analysis ==========================
def process_load_data(load, dt_runtimes, sim_runtimes, all_stats, all_runtime_data, num_nodes=None):
    """Helper function to process runtime data for a specific load.

    If `num_nodes` is provided, include it in the statistics record so
    cross-size tables can be constructed later.
    """
    # Calculate statistics
    dt_stats = {
        'num_nodes': num_nodes,
        'load': load,
        'algorithm': 'DT',
        'median': np.median(dt_runtimes),
        'q25': np.percentile(dt_runtimes, 25),
        'q75': np.percentile(dt_runtimes, 75),
        'q10': np.percentile(dt_runtimes, 10),
        'q90': np.percentile(dt_runtimes, 90),
        'mean': np.mean(dt_runtimes),
        'std': np.std(dt_runtimes),
        'min': np.min(dt_runtimes),
        'max': np.max(dt_runtimes),
        'n_trials': len(dt_runtimes)
    }
    
    sim_stats = {
        'num_nodes': num_nodes,
        'load': load,
        'algorithm': 'Simulation',
        'median': np.median(sim_runtimes),
        'q25': np.percentile(sim_runtimes, 25),
        'q75': np.percentile(sim_runtimes, 75),
        'q10': np.percentile(sim_runtimes, 10),
        'q90': np.percentile(sim_runtimes, 90),
        'mean': np.mean(sim_runtimes),
        'std': np.std(sim_runtimes),
        'min': np.min(sim_runtimes),
        'max': np.max(sim_runtimes),
        'n_trials': len(sim_runtimes)
    }
    
    all_stats.extend([dt_stats, sim_stats])
    
    # Store raw data for box plots
    for rt in dt_runtimes:
        all_runtime_data.append({'load': load, 'algorithm': 'DT', 'runtime': rt})
    for rt in sim_runtimes:
        all_runtime_data.append({'load': load, 'algorithm': 'Simulation', 'runtime': rt})

def analyze_cross_load_performance(output_dir, datapath):
    """
    Analyze performance across different load values by reading all CSV files
    and generating comparative statistics and plots.
    """
    # Find all runtime summary CSV files (preferred) or main CSV files
    pattern_summary = os.path.join(output_dir, "runtime_summary_*.csv")
    pattern_main = os.path.join(output_dir, "sp_only_test_*.csv")
    
    csv_files = glob.glob(pattern_summary)
    if not csv_files:
        csv_files = glob.glob(pattern_main)
    
    if not csv_files:
        print("No runtime CSV files found for cross-load analysis")
        return
    
    all_stats = []
    all_runtime_data = []
    
    for csv_file in csv_files:
        try:
            df = pd.read_csv(csv_file)
            if len(df) == 0:
                continue
            
            # Check if this is a runtime summary file or main file
            if 'load_streaming' in df.columns:
                # Runtime summary file - it may already include num_nodes per row
                for load in df['load_streaming'].unique():
                    load_data = df[df['load_streaming'] == load]
                    if len(load_data) < 2:
                        continue

                    # If num_nodes is present, process per (num_nodes, load)
                    if 'num_nodes' in load_data.columns:
                        for nn in sorted(load_data['num_nodes'].unique()):
                            grp = load_data[load_data['num_nodes'] == nn]
                            if len(grp) < 2:
                                continue
                            dt_runtimes = grp['runtime_dt'].values
                            sim_runtimes = grp['runtime_sim'].values
                            process_load_data(load, dt_runtimes, sim_runtimes, all_stats, all_runtime_data, num_nodes=int(nn))
                    else:
                        dt_runtimes = load_data['runtime_dt'].values
                        sim_runtimes = load_data['runtime_sim'].values
                        process_load_data(load, dt_runtimes, sim_runtimes, all_stats, all_runtime_data, num_nodes=None)
            else:
                # Main file - extract load from filename and process
                filename = os.path.basename(csv_file)
                parts = filename.split('_ls_')
                if len(parts) > 1:
                    load_str = parts[1].split('.csv')[0]
                    load = float(load_str)
                else:
                    continue
                
                # Get unique runtime values per f_case and try to read num_nodes from file
                unique_data = df.drop_duplicates(subset=['f_case', 'filename'])

                if len(unique_data) < 2:
                    continue

                dt_runtimes = unique_data['runtime_dt'].values
                sim_runtimes = unique_data['runtime_sim'].values

                # try to get num_nodes if available in the main CSV
                if 'num_nodes' in unique_data.columns:
                    try:
                        num_nodes = int(unique_data['num_nodes'].iloc[0])
                    except Exception:
                        num_nodes = None
                else:
                    num_nodes = None

                # Process this load's data (with optional num_nodes)
                process_load_data(load, dt_runtimes, sim_runtimes, all_stats, all_runtime_data, num_nodes=num_nodes)

                
        except Exception as e:
            print(f"Error processing {csv_file}: {e}")
            continue
    
    if not all_stats:
        print("No valid data found for cross-load analysis")
        return
    
    # Convert to DataFrames
    stats_df = pd.DataFrame(all_stats)
    runtime_df = pd.DataFrame(all_runtime_data)
    
    # Save statistics table
    stats_csv = os.path.join(output_dir, "cross_load_runtime_statistics.csv")
    stats_df.to_csv(stats_csv, index=False)
    print(f"Cross-load statistics saved to: {stats_csv}")

    
    # Print summary
    print("\n" + "="*80)
    print("CROSS-LOAD RUNTIME ANALYSIS SUMMARY")
    print("="*80)
    for load in sorted(stats_df['load'].unique()):
        load_stats = stats_df[stats_df['load'] == load]
        dt_stats = load_stats[load_stats['algorithm'] == 'DT'].iloc[0]
        sim_stats = load_stats[load_stats['algorithm'] == 'Simulation'].iloc[0]
        
        print(f"\nLoad Streaming: {load}")
        print(f"  DT Runtime:     {dt_stats['median']:.4f}s (median), {dt_stats['mean']:.4f}±{dt_stats['std']:.4f}s (mean±std)")
        print(f"  Sim Runtime:    {sim_stats['median']:.4f}s (median), {sim_stats['mean']:.4f}±{sim_stats['std']:.4f}s (mean±std)")
        print(f"  Median Speedup: {sim_stats['median']/dt_stats['median']:.1f}x")
        print(f"  Trials:         {dt_stats['n_trials']}")

# Run cross-load analysis if multiple CSV files exist
try:
    analyze_cross_load_performance(output_dir, datapath)
except Exception as e:
    print(f"Cross-load analysis failed: {e}")

print(f'Done')