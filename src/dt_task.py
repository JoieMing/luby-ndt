
import numpy as np
import tensorflow as tf
import scipy.sparse as sp
from typing import Optional, Dict, Tuple


def csr_to_edge_lists(A_csr):
    if not sp.isspmatrix_csr(A_csr):
        raise TypeError("A_csr must be a scipy.sparse.csr_matrix")
    A_csr = A_csr.tocsr()
    coo = A_csr.tocoo(copy=False)
    row_idx = coo.row.astype(np.int32)  # e
    col_idx = coo.col.astype(np.int32)  # i
    return row_idx, col_idx, A_csr.shape[0]


def make_b_ie_from_joint(
    b_init: np.ndarray,
    joint,
    src_idx: np.ndarray,
    dst_idx: np.ndarray,
    eps: float = 1e-12
) -> np.ndarray:
    E = b_init.shape[0]
    if sp is not None and sp.isspmatrix(joint):
        vals = joint.data
    else:
        vals = joint[src_idx, dst_idx]
    denom = np.maximum(b_init[dst_idx], eps)
    b_ie = np.clip(vals / denom, 0.0, 1.0).astype(np.float32)
    return b_ie


def build_transpose_map(row_idx, col_idx, E):
    """
    Given edge list (row_idx, col_idx) on a symmetric adjacency,
    returns an integer array trans_map where trans_map[k]
    is the index of the transposed edge (col_idx[k], row_idx[k]).

    Args:
        row_idx: np.ndarray [V], int32  (rows e)
        col_idx: np.ndarray [V], int32  (cols i)
        E: int, number of nodes/links
    Returns:
        trans_map: np.ndarray [V], int64, so that
                   for every k, (row_idx[trans_map[k]], col_idx[trans_map[k]]) == (col_idx[k], row_idx[k])
    """
    # Build a dict mapping (row,col) -> index
    pair_to_idx = {(int(r), int(c)): k for k, (r, c) in enumerate(zip(row_idx, col_idx))}
    # Now for each edge k, find its transpose index
    trans_map = np.empty_like(row_idx, dtype=np.int64)
    for k, (r, c) in enumerate(zip(row_idx, col_idx)):
        trans_map[k] = pair_to_idx.get((int(c), int(r)), -1)
    if np.any(trans_map < 0):
        missing = np.sum(trans_map < 0)
        print(f"[warn] {missing} edges had no transpose match.")
    return trans_map


@tf.function
def analytical_duty_cycle_round1_tf_adj_edges(
    b_init: tf.Tensor,
    src_idx: tf.Tensor,
    dst_idx: tf.Tensor,
    z: Optional[tf.Tensor] = None,
    b_ie_vals: Optional[tf.Tensor] = None,
    L: int = 100,
    eps: float = 1e-12,
    return_pwin: bool = False,
):
    b_init = tf.convert_to_tensor(b_init, tf.float32)
    src_idx = tf.convert_to_tensor(src_idx, tf.int32)
    dst_idx = tf.convert_to_tensor(dst_idx, tf.int32)
    E = tf.shape(b_init)[0]
    if z is None:
        z = tf.ones_like(b_init)
    else:
        z = tf.convert_to_tensor(z, dtype=b_init.dtype)
    z = tf.maximum(z, tf.cast(eps, z.dtype))

    if b_ie_vals is None:
        b_ie_vals = tf.gather(b_init, src_idx)  # independence
    else:
        b_ie_vals = tf.convert_to_tensor(b_ie_vals, dtype=b_init.dtype)

    L_val = tf.cast(tf.maximum(L, 1), tf.int32)
    l = tf.cast(tf.range(L_val)[None, :], b_init.dtype)        # [1,L]
    x_grid = (l / tf.cast(L_val, b_init.dtype)) * z[:, None]   # [E,L]
    x_edge = tf.gather(x_grid, dst_idx)                        # [K,L]

    z_src = tf.gather(z, src_idx)                              # [K]
    t = tf.clip_by_value(x_edge / z_src[:, None], 0.0, 1.0)    # [K,L]
    Fi = (1.0 - b_ie_vals[:, None]) + b_ie_vals[:, None] * t   # [K,L]

    log_Fi = tf.math.log(tf.clip_by_value(Fi, tf.cast(eps, Fi.dtype), 1.0))  # [K,L]

    sum_log = tf.math.unsorted_segment_sum(
        data=log_Fi,                    # [K,L]
        segment_ids=dst_idx,            # [K]
        num_segments=tf.cast(E, tf.int32)
    )  # [E,L]

    p_win = tf.reduce_mean(tf.exp(sum_log), axis=1)            # [E]
    duty = b_init * p_win

    if return_pwin:
        return duty, p_win
    return duty

@tf.function
def analytical_duty_cycle_Mrounds_tf_adj_edges(
    b_init: tf.Tensor,              
    src_idx: tf.Tensor,             
    dst_idx: tf.Tensor,             
    z: Optional[tf.Tensor] = None,  
    b_ie_vals_round1: Optional[tf.Tensor] = None,  
    M: int = 5,                     
    L: int = 100,                   
    eps: float = 1e-12,
    return_pwin: bool = False,
):
    b_init = tf.convert_to_tensor(b_init, tf.float32)
    src_idx = tf.convert_to_tensor(src_idx, tf.int32)
    dst_idx = tf.convert_to_tensor(dst_idx, tf.int32)
    E = tf.shape(b_init)[0]

    # z_e
    if z is None:
        z = tf.ones_like(b_init)
    else:
        z = tf.convert_to_tensor(z, dtype=b_init.dtype)
    z = tf.maximum(z, tf.cast(eps, z.dtype))

    if b_ie_vals_round1 is None:
        b_ie_vals_round1 = tf.gather(b_init, src_idx)  # [K]
    else:
        b_ie_vals_round1 = tf.convert_to_tensor(b_ie_vals_round1, dtype=b_init.dtype)

    L_val = tf.cast(tf.maximum(L, 1), tf.int32)
    l = tf.cast(tf.range(L_val)[None, :], b_init.dtype)        # [1,L]
    x_grid = (l / tf.cast(L_val, b_init.dtype)) * z[:, None]   # [E,L]
    x_edge = tf.gather(x_grid, dst_idx)                        # [K,L]
    
    z_src = tf.gather(z, src_idx)                              # [K]

    x_accum = tf.zeros_like(b_init)                # [E]  
    b_round_0 = tf.clip_by_value(b_init, 0.0, 1.0)
    b_m = b_round_0                                # [E]


    for m in range(1, M+1):
        if m == 1:
            b_used_edge = b_ie_vals_round1        # [K]
        else:
            b_used_edge = tf.gather(b_m, src_idx) # [K]

        t = tf.clip_by_value(x_edge / z_src[:, None], 0.0, 1.0)              # [K,L]
        Fi = (1.0 - b_used_edge[:, None]) + b_used_edge[:, None] * t         # [K,L]
        log_Fi = tf.math.log(tf.clip_by_value(Fi, tf.cast(eps, Fi.dtype), 1.0))

        sum_log = tf.math.unsorted_segment_sum(  # ∑_{i∈N(e)} log Fi(·)
            data=log_Fi,                         # [K,L]
            segment_ids=dst_idx,                 # [K]
            num_segments=tf.cast(E, tf.int32)
        )  # [E,L]

        p_win_m = tf.reduce_mean(tf.exp(sum_log), axis=1)                    # [E]

        x_accum = x_accum + b_m * p_win_m                                     # [E]

        if m == M:
            break
        def survive_next(b_current, p_win_current, use_conditional=False):
            # blockers[i] = 1 - b_used * p_win_current[i]
            if use_conditional:
                # m==1: b_used_edge = b^{(1)}_{i|e}
                b_used_edge_surv = b_ie_vals_round1  # [K]
            else:
                # m>1: b_used_edge = b^{(m)}_i
                b_used_edge_surv = tf.gather(b_current, src_idx)  # [K]

            blockers_src = 1.0 - tf.gather(p_win_current, src_idx) * b_used_edge_surv  # [K]
            log_blockers_src = tf.math.log(tf.clip_by_value(blockers_src, tf.cast(eps, blockers_src.dtype), 1.0))
            sum_log_blockers = tf.math.unsorted_segment_sum(
                data=log_blockers_src, segment_ids=dst_idx, num_segments=tf.cast(E, tf.int32)
            )  # [E]
            prod_blockers = tf.exp(sum_log_blockers)  # [E]
            b_next_local = b_current * (1.0 - p_win_current) * prod_blockers
            return tf.clip_by_value(b_next_local, 0.0, 1.0)

        if m == 1:
            b_next = survive_next(b_m, p_win_m, use_conditional=True)
        else:
            b_next = survive_next(b_m, p_win_m, use_conditional=False)

        b_m = b_next

    # —— duty —— #
    duty = x_accum  # [E]

    return duty


def compute_marginals_b(x_init, link_rates, lambda_e_vec, eps=1e-12, clip=True):
    """
    b_e = lambda_e / (x_init[e] * link_rates[e])
    """
    mu = np.asarray(x_init, np.float32) * np.asarray(link_rates, np.float32)
    b = np.asarray(lambda_e_vec, np.float32) / np.maximum(mu, eps)
    if clip:
        b = np.clip(b, 0.0, 1.0, out=b)
    return b, mu


def frechet_bounds_on_edges(b, row_idx, col_idx):
    """
    b: [E] marginals; row_idx=e, col_idx=i
    Returns L_vals, U_vals, p_ind_vals on the edge list (length V)
    """
    b_e = b[row_idx]
    b_i = b[col_idx]
    L_vals = np.maximum(0.0, b_e + b_i - 1.0).astype(np.float32)
    U_vals = np.minimum(b_e, b_i).astype(np.float32)
    p_ind_vals = (b_e * b_i).astype(np.float32)
    return L_vals, U_vals, p_ind_vals


def overlap_and_beta_from_lambda(lambda_mtx, lambda_e_vec, mu, row_idx, col_idx, eps=1e-12):
    """
    Build per-edge features aligned with the (row_idx=e, col_idx=i) edge list.
    lambda_mtx: CSR/COO with T_{e,i} on the same sparsity as A.
    Returns:
      T_vals, eta_to, eta_through, beta_to, beta_through, delta_log_mu
    """
    if not sp.isspmatrix(lambda_mtx):
        raise TypeError("lambda_mtx must be sparse")
    # Use the same (row,col) ordering as the adjacency for consistent indexing
    lam = lambda_mtx.tocsr()
    # Advanced indexing on CSR for edge order (row_idx, col_idx)
    T_vals = lam[row_idx, col_idx].A1.astype(np.float32)  # [V]

    lam_e = lambda_e_vec[row_idx]  # λ at row e
    lam_i = lambda_e_vec[col_idx]  # λ at col i
    mu_e  = mu[row_idx]            # μ at e
    mu_i  = mu[col_idx]            # μ at i

    eta_to       = np.clip(T_vals / np.maximum(lam_i, eps), 0.0, 1.0)   # T_{e,i} / λ_i (column-wise)
    eta_through  = np.clip(T_vals / np.maximum(lam_e, eps), 0.0, 1.0)   # T_{e,i} / λ_e (row-wise)

    beta_to      = np.clip(T_vals / np.maximum(mu_i,  eps), 0.0, 1.0)   # T_{e,i} / μ_i
    beta_through = np.clip(T_vals / np.maximum(mu_e,  eps), 0.0, 1.0)   # T_{e,i} / μ_e

    delta_log_mu = (np.log(mu_e + eps) - np.log(mu_i + eps)).astype(np.float32)

    return T_vals, eta_to.astype(np.float32), eta_through.astype(np.float32), \
           beta_to.astype(np.float32), beta_through.astype(np.float32), delta_log_mu


def symmetrize_edge_vector_same_size(values, row_idx, col_idx, E):
    """
    values: [V] per-directed-edge vector on a symmetric graph (both (e,i) and (i,e) exist)
    Returns: sym_values [V] where for each undirected pair {u,v}, both directions
             carry the average of the two directed values.
    """
    # Pair-id: {u,v} as unique id
    row64 = row_idx.astype(np.int64)
    col64 = col_idx.astype(np.int64)
    u = np.minimum(row64, col64)
    v = np.maximum(row64, col64)
    pair_id = u * np.int64(E) + v  # [V]

    # Reduce by pair-id (mean of the two directions)
    # Use numpy since we’re still in preprocessing; if you want TF gradients on values,
    # we can switch to tf.unsorted_segment_mean.
    _, inv = np.unique(pair_id, return_inverse=True)
    # Accumulate sums and counts per pair
    sums = np.bincount(inv, weights=values.astype(np.float64))
    cnts = np.bincount(inv)
    means = (sums / np.maximum(cnts, 1)).astype(np.float32)

    # Broadcast back to each directed edge
    return means[inv].astype(np.float32)


@tf.function
def postprocess_joint_from_delta_tf(
    L_vals, U_vals, p_ind_vals, Delta_sym_tf, eps=1e-12
):
    """
    TF2 version (fully differentiable).
    Vector form, all tensors length V.

      1) neutral = (p_ind - L) / (U - L), clipped to (0,1)
      2) logit(neutral) + Delta_sym_tf -> sigmoid -> pi_hat
      3) joint = L + (U - L) * pi_hat

    Args:
        L_vals, U_vals, p_ind_vals : numpy arrays or tensors of shape [V]
        Delta_sym_tf : tf.Tensor [V], from model forward pass
        eps : small constant for numerical stability

    Returns:
        joint_est_tf : tf.Tensor [V], symmetric joint estimates
    """
    # Convert numpy arrays → tf.float32 tensors
    L_tf = tf.convert_to_tensor(L_vals, dtype=tf.float32)
    U_tf = tf.convert_to_tensor(U_vals, dtype=tf.float32)
    p_ind_tf = tf.convert_to_tensor(p_ind_vals, dtype=tf.float32)
    Delta_sym_tf = tf.cast(Delta_sym_tf, tf.float32)

    eps_tf = tf.constant(eps, dtype=tf.float32)

    # Step 1: compute gap and neutral point
    gap_tf = tf.maximum(U_tf - L_tf, eps_tf)
    neutral_tf = tf.clip_by_value((p_ind_tf - L_tf) / gap_tf, 1e-6, 1.0 - 1e-6)

    # Step 2: logit(neutral) + Delta → sigmoid → pi_hat
    logit_neutral_tf = tf.math.log(neutral_tf) - tf.math.log1p(-neutral_tf)
    pi_hat_tf = tf.sigmoid(logit_neutral_tf + Delta_sym_tf)

    # Step 3: joint = L + gap * pi_hat
    joint_est_tf = L_tf + gap_tf * pi_hat_tf
    return joint_est_tf


class EdgeMLP(tf.keras.Model):
    def __init__(self, hidden_dims=(64, 64), dropout=0.0, **kwargs):
        super().__init__(**kwargs)
        self.layers_ = []
        for h in hidden_dims:
            self.layers_.append(tf.keras.layers.Dense(h, activation='relu'))
            if dropout and dropout > 0:
                self.layers_.append(tf.keras.layers.Dropout(dropout))
        self.out = tf.keras.layers.Dense(1, activation='sigmoid')

    @tf.function
    def call(self, edge_inputs: tf.Tensor) -> tf.Tensor:
        x = edge_inputs
        for lyr in self.layers_:
            x = lyr(x)
        return self.out(x)  # [K,1]


def build_edge_inputs_from_arrays(
    src_idx: np.ndarray,
    dst_idx: np.ndarray,
    node_features: Dict[str, np.ndarray],
    edge_features: Dict[str, np.ndarray],
    A_csr,
) -> np.ndarray:
    E = A_csr.shape[0]
    K = len(src_idx)
    def get_node(name):
        return node_features[name].astype(np.float32) if name in node_features else np.zeros((E,), np.float32)
    lam = get_node('lambda_e')
    x0  = get_node('x_init')
    def pick_edge(mtx, s, d):
        if sp.isspmatrix(mtx):
            return mtx.tocsr()[s, d].astype(np.float32)
        else:
            return mtx[s, d].astype(np.float32)
    def get_edge(name):
        if name in edge_features:
            return pick_edge(edge_features[name], src_idx, dst_idx)
        else:
            return np.zeros((K,), np.float32)
    T_ie = get_edge('T_ie')
    T_ei = get_edge('T_ei')
    eta_toe = get_edge('eta_toe')
    eta_through = get_edge('eta_through')
    cols = [
        T_ie, T_ei,
        eta_toe, eta_through,
        lam[src_idx], lam[dst_idx],
        x0[src_idx],  x0[dst_idx],
    ]
    X_edge = np.stack(cols, axis=1)
    return X_edge.astype(np.float32)


def gnn_and_dt_round1_pipeline(
    A_csr,
    lambda_e_vec: np.ndarray,
    x_init: np.ndarray,
    edge_mats: Dict[str, np.ndarray],
    L: int = 128,
    hidden_dims=(64,64),
    dropout=0.0,
    return_pwin: bool = True,
):
    src_idx, dst_idx, E = csr_to_edge_lists(A_csr)
    node_feats = {'lambda_e': lambda_e_vec, 'x_init': x_init}
    X_edge = build_edge_inputs_from_arrays(src_idx, dst_idx, node_feats, edge_mats, A_csr)  # [K, F]
    edge_mlp = EdgeMLP(hidden_dims=hidden_dims, dropout=dropout)
    b_ie_vals = tf.squeeze(edge_mlp(tf.convert_to_tensor(X_edge, tf.float32)), axis=-1)  # [K]
    duty_tf, pwin_tf = analytical_duty_cycle_round1_tf_adj_edges(
        b_init=tf.convert_to_tensor(x_init, tf.float32),
        src_idx=tf.convert_to_tensor(src_idx, tf.int32),
        dst_idx=tf.convert_to_tensor(dst_idx, tf.int32),
        z=None,
        b_ie_vals=b_ie_vals,
        L=L,
        eps=1e-12,
        return_pwin=return_pwin,
    )
    return duty_tf, pwin_tf, b_ie_vals


def mlp_sec24_forward_with_symmetry(
    A_csr,               # symmetric adjacency (rows=e, cols=i)
    lambda_mtx,          # CSR with T_{e,i} on same sparsity
    x_init, link_rates, lambda_e_vec,
    edge_mlp,            # tf.keras model: input [V,F] -> [V,1] via sigmoid
    eps=1e-12
):
    import numpy as np
    import scipy.sparse as sp
    import tensorflow as tf

    # 1) Edge list from A (row=e, col=i)
    row_idx, col_idx, E = csr_to_edge_lists(A_csr)         # [V], [V], E
    V = len(row_idx)
    trans_map = build_transpose_map(row_idx, col_idx, E)   # [V]
    trans_map_tf = tf.constant(trans_map, dtype=tf.int32)

    # 2) Marginals & μ
    b, mu = compute_marginals_b(x_init, link_rates, lambda_e_vec, eps=eps, clip=True)  # b:[E], mu:[E]

    # 3) Frechet bounds + p_ind on edges (numpy preprocessing OK)
    L_vals, U_vals, p_ind_vals = frechet_bounds_on_edges(b, row_idx, col_idx)          # [V] each

    # 4) Overlap/beta features + Δ log μ (aligned with (row=e, col=i))
    T_vals, eta_to, eta_through, beta_to, beta_through, dlogmu = \
        overlap_and_beta_from_lambda(lambda_mtx, lambda_e_vec, mu, row_idx, col_idx, eps=eps)

    # 5) Gate and assemble edge features [V,F]
    gate = np.sqrt(eta_to * eta_through).astype(np.float32)    # [V]
    feats_cols = [
        T_vals.astype(np.float32),
        eta_to.astype(np.float32), eta_through.astype(np.float32),
        beta_to.astype(np.float32), beta_through.astype(np.float32),
        dlogmu.astype(np.float32),
    ]
    X_edge = np.stack(feats_cols, axis=1).astype(np.float32)   # [V, F]

    # 6) MLP forward in TF
    X_edge_tf = tf.convert_to_tensor(X_edge, tf.float32)       # [V,F]
    scores_tf = tf.squeeze(edge_mlp(X_edge_tf), axis=-1)       # [V] in (0,1) if last act sigmoid, else arbitrary
    # If your MLP already outputs (0,1), this is fine; if you want raw logits, remove sigmoid in the model.

    # 7) Δ (directed) and symmetry enforcement via transpose map
    gate_tf = tf.convert_to_tensor(gate, tf.float32)           # [V]
    Delta_ie = gate_tf * scores_tf                              # [V]
    Delta_sym_tf = 0.5 * (Delta_ie + tf.gather(Delta_ie, trans_map_tf))  # [V]

    # 8) Symmetric joint via neutral logit shift (all TF, differentiable)
    joint_est_tf = postprocess_joint_from_delta_tf(
        L_vals, U_vals, p_ind_vals, Delta_sym_tf, eps=eps
    )  # [V]

    # 9) Column-wise conditionals: divide by b[col] (marginal of column i)
    b_tf = tf.convert_to_tensor(b, tf.float32)                              # [E]
    col_idx_tf = tf.convert_to_tensor(col_idx, tf.int32)                    # [V]
    b_col_tf = tf.gather(b_tf, col_idx_tf)                                  # [V]
    b_ie_tf = tf.clip_by_value(joint_est_tf / tf.maximum(b_col_tf, eps), 0.0, 1.0)  # [V]

    # 10) Optional sparse matrices for inspection (non-differentiable; CPU only)
    joint_est_np = joint_est_tf.numpy()
    b_ie_np = b_ie_tf.numpy()
    joint_est_coo = sp.coo_matrix((joint_est_np, (row_idx, col_idx)), shape=(E, E))
    b_ie_coo      = sp.coo_matrix((b_ie_np,      (row_idx, col_idx)), shape=(E, E))

    out = {
        "row_idx": row_idx, "col_idx": col_idx, "E": E,
        "X_edge": X_edge, "scores_tf": scores_tf,
        "Delta_ie_tf": Delta_ie, "Delta_sym_tf": Delta_sym_tf,
        "joint_est_tf": joint_est_tf, "b_ie_tf": b_ie_tf, "b_tf": b_tf,
        "joint_est_coo": joint_est_coo, "b_ie_coo": b_ie_coo,
        "b": b, "mu": mu, "L_vals": L_vals, "U_vals": U_vals, "p_ind_vals": p_ind_vals,
    }
    return out
