#!/usr/bin/env python3
"""
Graph and TSP Optimization Utilities

Provides GPU-accelerated (CuPy when available) TSP solvers and graph construction
algorithms for robot trajectory optimization. Includes k-NN, MST-based neighbor
graphs, greedy TSP with robot cost, and dynamic programming for IK selection.

Functions consolidated from:
- fk_gtsp_gpu_claude2.py: Graph construction and TSP solving
"""

import sys
from typing import List, Dict, Tuple

import numpy as np

# GPU acceleration (optional)
try:
    import cupy as cp
    GPU_AVAILABLE = True
except ImportError:
    cp = None
    GPU_AVAILABLE = False


# ============================================================================
# Graph Construction
# ============================================================================

def build_neighbors_knn(target_coords: np.ndarray, k: int) -> List[List[int]]:
    """
    Build k-nearest neighbor graph

    Args:
        target_coords: (M, 3) array of target positions
        k: Number of nearest neighbors per node

    Returns:
        nbrs: List of M neighbor lists, each containing indices of k nearest neighbors

    Example:
        >>> coords = np.random.rand(100, 3)
        >>> nbrs = build_neighbors_knn(coords, k=5)
        >>> print(len(nbrs[0]))  # First node has 5 neighbors
        5
    """
    M = target_coords.shape[0]
    nbrs = [[] for _ in range(M)]
    # Distance matrix
    dif = target_coords[:, None, :] - target_coords[None, :, :]
    dist = np.linalg.norm(dif, axis=-1)
    np.fill_diagonal(dist, np.inf)
    for i in range(M):
        idx = np.argsort(dist[i])[:max(1, min(k, M-1))]
        nbrs[i] = idx.tolist()
    return nbrs


def build_neighbors_auto(target_coords: np.ndarray) -> List[List[int]]:
    """
    Build adaptive neighbor graph with distance jump ratio heuristic + MST reinforcement

    Automatically determines number of neighbors per node based on distance distribution
    and adds MST edges to ensure connectivity.

    Args:
        target_coords: (M, 3) array of target positions

    Returns:
        nbrs: List of M neighbor lists with adaptive neighbor counts

    Algorithm:
        1. For each node, find neighbors based on distance jump ratio
        2. Compute MST (Prim's algorithm) for graph connectivity
        3. Add MST edges to neighbor lists
        4. Ensure each node has at least one neighbor

    Example:
        >>> coords = np.random.rand(50, 3)
        >>> nbrs = build_neighbors_auto(coords)
        >>> print(min(len(n) for n in nbrs))  # All nodes have neighbors
        1
    """
    M = target_coords.shape[0]
    if M == 0:
        return []

    dif = target_coords[:, None, :] - target_coords[None, :, :]
    dist = np.linalg.norm(dif, axis=-1)
    np.fill_diagonal(dist, np.inf)

    # 1) Distance jump ratio-based neighbors
    nbrs = [[] for _ in range(M)]
    for i in range(M):
        order = np.argsort(dist[i])  # Nearest first
        di = dist[i, order]
        m = np.isfinite(di).sum()
        if m == 0:
            nbrs[i] = []
            continue
        di = di[:m]; ord_use = order[:m]

        if m == 1:
            cand = [int(ord_use[0])]
        else:
            ratios = di[1:] / np.maximum(di[:-1], 1e-12)  # Adjacent distance ratios
            j_star = int(np.argmax(ratios))  # Largest jump
            cut = max(0, j_star)  # Ensure at least 1 neighbor
            cand = ord_use[:cut+1].tolist()

        if len(cand) == 0 and m > 0:  # Emergency guard
            cand = [int(ord_use[0])]
        nbrs[i] = cand

    # 2) MST (Prim's algorithm)
    visited = np.zeros(M, dtype=bool)
    if M > 0:
        visited[0] = True
    edges = []
    for _ in range(M - 1):
        in_nodes = np.where(visited)[0]
        out_nodes = np.where(~visited)[0]
        if out_nodes.size == 0:
            break

        D_sub = dist[in_nodes, :][:, out_nodes]
        if D_sub.size == 0:
            break  # Disconnected graph

        min_idx_flat = np.argmin(D_sub)
        min_idx_in, min_idx_out = np.unravel_index(min_idx_flat, D_sub.shape)
        u, v = in_nodes[min_idx_in], out_nodes[min_idx_out]
        w = D_sub[min_idx_in, min_idx_out]

        if w == np.inf:
            break  # Cannot connect
        visited[v] = True
        edges.append((u, v))

    # 3) Add MST edges to neighbor lists
    for u, v in edges:
        if v not in nbrs[u]:
            nbrs[u].append(v)
        if u not in nbrs[v]:
            nbrs[v].append(u)

    # 4) Emergency guard: if node has no neighbors, add nearest
    for i in range(M):
        if len(nbrs[i]) == 0 and M > 1:
            nbrs[i] = [int(np.argmin(dist[i]))]

    return nbrs


# ============================================================================
# TSP Solvers
# ============================================================================

def build_visit_order_robot_cost(
    clusters: List[Dict],
    nbrs: List[List[int]],
    lam_rot: float,
    tool_z: float
) -> List[int]:
    """
    Build visit order using greedy TSP with robot motion cost

    Starting from the target closest to origin (0,0,0), greedily selects the next
    target with minimum robot motion cost among neighbors (with fallback to Euclidean
    distance if no unvisited neighbors remain).

    Args:
        clusters: List of cluster dictionaries with keys "q", "R", "p", "Q", "target"
        nbrs: Neighbor list for each cluster
        lam_rot: Rotation cost weight
        tool_z: Tool length offset in meters

    Returns:
        order: List of cluster indices in visit order

    Algorithm:
        1. Start from cluster closest to origin
        2. At each step, choose unvisited neighbor with minimum pair_cost
        3. If no unvisited neighbors, choose unvisited node with minimum Euclidean distance
        4. Repeat until all nodes visited

    Example:
        >>> order = build_visit_order_robot_cost(clusters, nbrs, lam_rot=1.0, tool_z=0.0)
        >>> print(len(order))
        100
    """
    M = len(clusters)
    if M == 0:
        return []

    # Target coordinates (M,3) and distance matrix
    target_coords = np.stack([c["target"] for c in clusters], axis=0)
    dif = target_coords[:, None, :] - target_coords[None, :, :]
    dist_mat = np.linalg.norm(dif, axis=-1)
    np.fill_diagonal(dist_mat, np.inf)

    # 1) Cost computation helper (with caching)
    edge_cache = {}
    def pair_cost(a: int, b: int) -> float:
        key = tuple(sorted((a, b)))
        if key in edge_cache:
            return edge_cache[key]

        A = clusters[a]; B = clusters[b]
        C = pair_cost_matrix_joint_mid(
            A["q"], A["R"], A["p"], A["Q"],
            B["q"], B["R"], B["p"], B["Q"],
            lam_rot=lam_rot, tool_z=tool_z
        )
        cost = float(np.min(C))
        edge_cache[key] = cost
        return cost

    # 2) Starting point (closest to origin)
    start = int(np.argmin(np.linalg.norm(target_coords, axis=1)))

    order = [start]
    visited = np.zeros(M, dtype=bool)
    visited[start] = True
    cur = start

    # 3) Greedy search
    for _ in range(M - 1):
        best_cost, best_j = np.inf, -1

        # Search within neighbors (robot cost-based)
        cand = [v for v in nbrs[cur] if not visited[v]]

        if cand:
            # Normal case: compute actual robot cost for neighbor candidates only
            for j in cand:
                c = pair_cost(cur, j)
                if c < best_cost:
                    best_cost, best_j = c, j
        else:
            # Fallback: select node with shortest Euclidean distance
            unvisited = np.where(~visited)[0]
            if unvisited.size == 0:
                break
            dist_row = dist_mat[cur].copy()
            dist_row[visited] = np.inf
            best_j = int(np.argmin(dist_row))
            # No pair_cost call here (pure distance-based)

        if best_j == -1:
            print(f"[WARN] TSP greedy search stuck at node {cur}. Graph may be disconnected.", file=sys.stderr)
            break

        visited[best_j] = True
        order.append(best_j)
        cur = best_j

    return order


def choose_ik_given_order(
    clusters: List[Dict],
    order: List[int],
    lam_rot: float,
    tool_z: float,
) -> Tuple[Dict[int, int], float]:
    """
    Optimize IK selection for given visit order using Dynamic Programming

    For a fixed visit order, selects the best IK solution at each waypoint to minimize
    total trajectory cost using vectorized DP.

    Args:
        clusters: List of cluster dictionaries
        order: Visit order (list of cluster indices)
        lam_rot: Rotation cost weight
        tool_z: Tool length offset in meters

    Returns:
        picked: Dictionary mapping cluster index → selected IK index
        total_cost: Total trajectory cost

    Algorithm:
        1. Pre-compute pairwise cost matrices for consecutive clusters
        2. Run vectorized DP: dp[i] = min cost to reach configuration i
        3. Backtrack to find optimal IK selection

    Example:
        >>> picked, cost = choose_ik_given_order(clusters, order, lam_rot=1.0, tool_z=0.0)
        >>> print(f"Total cost: {cost:.2f}")
        Total cost: 42.15
        >>> print(f"Selected IK for cluster 0: {picked[0]}")
        Selected IK for cluster 0: 3
    """
    n = len(order)
    if n == 0:
        return {}, 0.0

    # Pre-compute cost matrices for each segment
    cost_mats: List[np.ndarray] = []
    for t in range(n - 1):
        ia, ib = order[t], order[t + 1]
        A = clusters[ia]; B = clusters[ib]
        A_q, A_R, A_p, A_Q = A["q"], A["R"], A["p"], A["Q"]
        B_q, B_R, B_p, B_Q = B["q"], B["R"], B["p"], B["Q"]

        C = pair_cost_matrix_joint_mid(
            A_q, A_R, A_p, A_Q,
            B_q, B_R, B_p, B_Q,
            lam_rot=lam_rot,
            tool_z=tool_z,
        )
        cost_mats.append(C)

    # DP (fully vectorized)
    Sa0 = clusters[order[0]]["q"].shape[0]
    dp = np.zeros(Sa0, dtype=np.float64)
    back = [np.full((Sa0,), -1, dtype=np.int32)]

    for t in range(n - 1):
        C = cost_mats[t]
        Sa, Sb = C.shape

        # Vectorized: tmp[i,j] = dp[i] + C[i,j]
        tmp = dp[:, None] + C  # (Sa, Sb)
        dp = np.min(tmp, axis=0)  # (Sb,)
        arg = np.argmin(tmp, axis=0).astype(np.int32)  # (Sb,)
        back.append(arg)

    # Backtracking
    j_star = int(np.argmin(dp))
    total_cost = float(dp[j_star])

    picked_local = [None] * n
    picked_local[-1] = j_star
    for t in range(n - 2, -1, -1):
        i_star = int(back[t + 1][picked_local[t + 1]])
        picked_local[t] = i_star

    # Mapping
    picked = {}
    for t, cidx in enumerate(order):
        picked[cidx] = int(picked_local[t])

    return picked, total_cost


# ============================================================================
# Cost Computation
# ============================================================================

def pair_cost_matrix_joint_mid(
    A_q: np.ndarray, A_R: np.ndarray, A_p: np.ndarray, A_Q: np.ndarray,
    B_q: np.ndarray, B_R: np.ndarray, B_p: np.ndarray, B_Q: np.ndarray,
    lam_rot: float,
    tool_z: float,
) -> np.ndarray:
    """
    Compute pairwise motion cost matrix with joint-space interpolation midpoint

    For each pair of IK solutions (A[i], B[j]), computes cost via intermediate
    configuration at joint-space midpoint:
        cost[i,j] = d(A[i], mid) + d(mid, B[j])

    Uses GPU acceleration (CuPy) when available for large matrices.

    Args:
        A_q: (SA, DOF) joint configurations for cluster A
        A_R: (SA, 3, 3) end-effector rotations for A
        A_p: (SA, 3) end-effector positions for A
        A_Q: (SA, 4) end-effector quaternions for A (not used, kept for API)
        B_q: (SB, DOF) joint configurations for cluster B
        B_R: (SB, 3, 3) end-effector rotations for B
        B_p: (SB, 3) end-effector positions for B
        B_Q: (SB, 4) end-effector quaternions for B (not used, kept for API)
        lam_rot: Rotation cost weight
        tool_z: Tool length offset in meters

    Returns:
        cost: (SA, SB) cost matrix

    Cost formula:
        For each pair (A[i], B[j]):
        1. q_mid = 0.5 * (A[i] + B[j])
        2. Compute FK(q_mid) → (R_mid, p_mid)
        3. cost = ||p_A - p_mid|| + λ*angle(R_A, R_mid) +
                  ||p_mid - p_B|| + λ*angle(R_mid, R_B)

    Note:
        Automatically uses GPU (CuPy) for SA*SB > 1000 if available.
        Uses batching to manage memory.
    """
    # Import kinematics utilities
    from common.kinematics_utils import fk_batch, rot_angle_ignore_tool_yaw

    SA, SB = A_q.shape[0], B_q.shape[0]

    # Swap to iterate over smaller dimension
    swap = False
    if SB < SA:
        A_q, B_q = B_q, A_q
        A_R, B_R = B_R, A_R
        A_p, B_p = B_p, A_p
        A_Q, B_Q = B_Q, A_Q
        SA, SB = SB, SA
        swap = True

    # Use GPU if available and problem is large enough
    use_gpu = GPU_AVAILABLE and (SA * SB > 1000)

    if use_gpu:
        # Transfer data to GPU
        A_p_gpu = cp.asarray(A_p)
        B_p_gpu = cp.asarray(B_p)
        A_R_gpu = cp.asarray(A_R)
        B_R_gpu = cp.asarray(B_R)

    cost = np.empty((SA, SB), dtype=np.float64)

    # Dynamic batch size (balance memory and performance)
    if SA * SB > 10000:
        batch_size = min(32, SA)
    else:
        batch_size = SA  # Small: process all at once

    for batch_start in range(0, SA, batch_size):
        batch_end = min(batch_start + batch_size, SA)
        batch_slice = slice(batch_start, batch_end)
        n_batch = batch_end - batch_start

        # Compute joint midpoints (vectorized)
        A_q_batch = A_q[batch_slice]
        q_mid_all = 0.5 * (A_q_batch[:, None, :] + B_q[None, :, :])
        q_mid_flat = q_mid_all.reshape(-1, A_q.shape[1])

        # FK computation (CPU batch)
        Rm, pm = fk_batch(q_mid_flat, tool_z)
        Rm = Rm.reshape(n_batch, SB, 3, 3)
        pm = pm.reshape(n_batch, SB, 3)

        if use_gpu:
            # GPU distance computation
            pm_gpu = cp.asarray(pm)
            Rm_gpu = cp.asarray(Rm)
            A_p_batch_gpu = A_p_gpu[batch_slice]
            A_R_batch_gpu = A_R_gpu[batch_slice]

            # d(Ti, Tmid) - GPU
            d1_pos = cp.linalg.norm(pm_gpu - A_p_batch_gpu[:, None, :], axis=2)

            # Rotation angle - GPU
            za_batch = A_R_batch_gpu[:, None, :, 2]  # (n_batch, 1, 3)
            zm = Rm_gpu[:, :, :, 2]  # (n_batch, SB, 3)
            dots1 = cp.sum(za_batch * zm, axis=2)
            dots1 = cp.clip(dots1, -1.0, 1.0)
            th1 = cp.arccos(dots1)

            # d(Tmid, Tj) - GPU
            d2_pos = cp.linalg.norm(B_p_gpu[None, :, :] - pm_gpu, axis=2)

            # Rotation angle - GPU
            zb = B_R_gpu[None, :, :, 2]  # (1, SB, 3)
            dots2 = cp.sum(zm * zb, axis=2)
            dots2 = cp.clip(dots2, -1.0, 1.0)
            th2 = cp.arccos(dots2)

            # Total cost - GPU
            cost_batch = d1_pos + lam_rot * th1 + d2_pos + lam_rot * th2

            # Copy to CPU
            cost[batch_slice, :] = cp.asnumpy(cost_batch)
        else:
            # CPU version (vectorized)
            A_p_batch = A_p[batch_slice]
            A_R_batch = A_R[batch_slice]

            d1_pos = np.linalg.norm(pm - A_p_batch[:, None, :], axis=2)
            th1 = rot_angle_ignore_tool_yaw(A_R_batch[:, None, :, :], Rm)

            d2_pos = np.linalg.norm(B_p[None, :, :] - pm, axis=2)
            th2 = rot_angle_ignore_tool_yaw(Rm, B_R[None, :, :, :])

            cost[batch_slice, :] = d1_pos + lam_rot * th1 + d2_pos + lam_rot * th2

    if swap:
        cost = cost.T

    return cost
