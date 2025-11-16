#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
fk_gtsp.py (로직 복원 버전)
- 여러 타깃(viewpoint)과 각 타깃의 다수 IK 해 중에서,
  방문 순서와 각 지점의 IK를 동시에 선택하여
  EE 경로 길이(위치 거리 + lam_rot*회전각)를 최소화.

[복원된 로직 1] 이웃 생성 (--knn 0):
  - '거리 점프 비율' + MST 보강 (이전 코드 방식)

[복원된 로직 2] TSP 순서 결정:
  - '실제 로봇 비용' (IK 전수비교 최소값) 기반 그리디 (이전 코드 방식)
  - (유지) 비용 계산: "조인트-보간 중간점 1개" FK (새 코드 방식)
  - (유지) IK 선택: 고정 순서 내에서 DP로 전역 최적화 (새 코드 방식)

Requirements:
    numpy, h5py
Optional:
    numba (있으면 FK/쿼터니언 변환 JIT 가속)
"""

import argparse
import math
import sys
from typing import Dict, List, Tuple, Optional

import numpy as np
import h5py

import time

# -----------------------------
# Numba 옵션 (변경 없음)
# -----------------------------
try:
    from numba import njit
    NUMBA_AVAILABLE = True
except Exception:
    NUMBA_AVAILABLE = False
    def njit(*args, **kwargs):
        def deco(fn):
            return fn
        return deco


# =============================
# 수학 유틸 (변경 없음)
# =============================
def rot_to_quat_batch_np(R: np.ndarray) -> np.ndarray:
    """회전행렬들(R: (N,3,3))을 쿼터니언들(Q: (N,4), (w,x,y,z))로 변환."""
    N = R.shape[0]
    Q = np.empty((N, 4), dtype=np.float64)
    for k in range(N):
        r = R[k]
        tr = r[0, 0] + r[1, 1] + r[2, 2]
        if tr > 0.0:
            t = math.sqrt(tr + 1.0) * 2.0
            qw = 0.25 * t
            qx = (r[2, 1] - r[1, 2]) / t
            qy = (r[0, 2] - r[2, 0]) / t
            qz = (r[1, 0] - r[0, 1]) / t
        else:
            if r[0, 0] > r[1, 1] and r[0, 0] > r[2, 2]:
                t = math.sqrt(1.0 + r[0, 0] - r[1, 1] - r[2, 2]) * 2.0
                qw = (r[2, 1] - r[1, 2]) / t
                qx = 0.25 * t
                qy = (r[0, 1] + r[1, 0]) / t
                qz = (r[0, 2] + r[2, 0]) / t
            elif r[1, 1] > r[2, 2]:
                t = math.sqrt(1.0 - r[0, 0] + r[1, 1] - r[2, 2]) * 2.0
                qw = (r[0, 2] - r[2, 0]) / t
                qx = (r[0, 1] + r[1, 0]) / t
                qy = 0.25 * t
                qz = (r[1, 2] + r[2, 1]) / t
            else:
                t = math.sqrt(1.0 - r[0, 0] - r[1, 1] + r[2, 2]) * 2.0
                qw = (r[1, 0] - r[0, 1]) / t
                qx = (r[0, 2] + r[2, 0]) / t
                qy = (r[1, 2] + r[2, 1]) / t
                qz = 0.25 * t
        n = math.sqrt(qw * qw + qx * qx + qy * qy + qz * qz)
        Q[k, 0] = qw / n
        Q[k, 1] = qx / n
        Q[k, 2] = qy / n
        Q[k, 3] = qz / n
    return Q


if NUMBA_AVAILABLE:
    @njit(cache=True, fastmath=True)
    def rot_to_quat_batch(R):
        N = R.shape[0]
        Q = np.empty((N, 4), np.float64)
        for k in range(N):
            r = R[k]
            tr = r[0, 0] + r[1, 1] + r[2, 2]
            if tr > 0.0:
                t = math.sqrt(tr + 1.0) * 2.0
                qw = 0.25 * t
                qx = (r[2, 1] - r[1, 2]) / t
                qy = (r[0, 2] - r[2, 0]) / t
                qz = (r[1, 0] - r[0, 1]) / t
            else:
                if r[0, 0] > r[1, 1] and r[0, 0] > r[2, 2]:
                    t = math.sqrt(1.0 + r[0, 0] - r[1, 1] - r[2, 2]) * 2.0
                    qw = (r[2, 1] - r[1, 2]) / t
                    qx = 0.25 * t
                    qy = (r[0, 1] + r[1, 0]) / t
                    qz = (r[0, 2] + r[2, 0]) / t
                elif r[1, 1] > r[2, 2]:
                    t = math.sqrt(1.0 - r[0, 0] + r[1, 1] - r[2, 2]) * 2.0
                    qw = (r[0, 2] - r[2, 0]) / t
                    qx = (r[0, 1] + r[1, 0]) / t
                    qy = 0.25 * t
                    qz = (r[1, 2] + r[2, 1]) / t
                else:
                    t = math.sqrt(1.0 - r[0, 0] - r[1, 1] + r[2, 2]) * 2.0
                    qw = (r[1, 0] - r[0, 1]) / t
                    qx = (r[0, 2] + r[2, 0]) / t
                    qy = (r[1, 2] + r[2, 1]) / t
                    qz = 0.25 * t
            n = math.sqrt(qw * qw + qx * qx + qy * qy + qz * qz)
            Q[k, 0] = qw / n
            Q[k, 1] = qx / n
            Q[k, 2] = qy / n
            Q[k, 3] = qz / n
        return Q
else:
    def rot_to_quat_batch(R):
        return rot_to_quat_batch_np(R)


# =============================
# UR20 예시 DH로 FK (배치) (변경 없음)
# =============================
# 표준 DH 파라미터(예시). 실제 장비 파라미터로 바꾸셔도 됩니다.
_A = np.array([0.0, -0.612, -0.5723, 0.0, 0.0, 0.0], dtype=np.float64)
_D = np.array([0.1807, 0.0, 0.0, 0.163941, 0.1157, 0.0922], dtype=np.float64)
_AL= np.array([np.pi/2, 0.0, 0.0, np.pi/2, -np.pi/2, 0.0], dtype=np.float64)

if NUMBA_AVAILABLE:
    @njit(cache=True, fastmath=True)
    def _fk_single(q, tool_z: float):
        T = np.eye(4, dtype=np.float64)
        for i in range(6):
            th = q[i]
            ca = math.cos(_AL[i]); sa = math.sin(_AL[i])
            ct = math.cos(th);     st = math.sin(th)
            # A_i
            A = np.empty((4,4), np.float64)
            A[0,0]=ct;   A[0,1]=-st*ca; A[0,2]= st*sa; A[0,3]=_A[i]*ct
            A[1,0]=st;   A[1,1]= ct*ca; A[1,2]=-ct*sa; A[1,3]=_A[i]*st
            A[2,0]=0.0;  A[2,1]= sa;    A[2,2]= ca;    A[2,3]=_D[i]
            A[3,0]=0.0;  A[3,1]= 0.0;   A[3,2]= 0.0;   A[3,3]=1.0
            T = T @ A
        if tool_z != 0.0:
            T[0,3] += T[0,2]*tool_z
            T[1,3] += T[1,2]*tool_z
            T[2,3] += T[2,2]*tool_z
        return T

    @njit(cache=True, fastmath=True, parallel=True)
    def fk_batch(qs, tool_z: float):
        N = qs.shape[0]
        R = np.empty((N,3,3), np.float64)
        p = np.empty((N,3),   np.float64)
        for k in range(N):
            T = _fk_single(qs[k], tool_z)
            R[k,0,0]=T[0,0]; R[k,0,1]=T[0,1]; R[k,0,2]=T[0,2]
            R[k,1,0]=T[1,0]; R[k,1,1]=T[1,1]; R[k,1,2]=T[1,2]
            R[k,2,0]=T[2,0]; R[k,2,1]=T[2,1]; R[k,2,2]=T[2,2]
            p[k,0]=T[0,3]; p[k,1]=T[1,3]; p[k,2]=T[2,3]
        return R, p
else:
    def _fk_single_np(q: np.ndarray, tool_z: float) -> np.ndarray:
        T = np.eye(4, dtype=np.float64)
        for i in range(6):
            th = q[i]
            ca = np.cos(_AL[i]); sa = np.sin(_AL[i])
            ct = np.cos(th);     st = np.sin(th)
            A = np.array([
                [ct,   -st*ca,  st*sa, _A[i]*ct],
                [st,    ct*ca, -ct*sa, _A[i]*st],
                [0.0,     sa,      ca, _D[i]],
                [0.0,    0.0,     0.0, 1.0]
            ], dtype=np.float64)
            T = T @ A
        if tool_z != 0.0:
            T[:3, 3] += T[:3, 2] * tool_z
        return T

    def fk_batch(qs: np.ndarray, tool_z: float):
        N = qs.shape[0]
        R = np.empty((N,3,3), dtype=np.float64)
        p = np.empty((N,3),   dtype=np.float64)
        for k in range(N):
            T = _fk_single_np(qs[k], tool_z)
            R[k] = T[:3,:3]
            p[k] = T[:3, 3]
        return R, p


# =============================
# 비용 계산(조인트-보간 중간점 1개) (변경 없음)
# =============================
def rotation_angle_from_quats(Qa: np.ndarray, Qb: np.ndarray) -> np.ndarray:
    """쿼터니언 배열 간 각도: 2*arccos(|<qa,qb>|). Qa, Qb shape must match on leading dims."""
    dots = np.abs(np.sum(Qa * Qb, axis=-1))
    dots = np.clip(dots, 0.0, 1.0)
    return 2.0 * np.arccos(dots)

def rot_angle_ignore_tool_yaw(Ra: np.ndarray, Rb: np.ndarray) -> np.ndarray:
    """
    Ra, Rb: (..., 3, 3)
    반환: (...,) — 두 자세의 '툴 z축' 사이각 (yaw 무시, tilt만)
    """
    # 각 회전행렬의 z축 단위벡터
    za = Ra[..., :, 2]   # (..., 3)
    zb = Rb[..., :, 2]   # (..., 3)
    dots = np.sum(za * zb, axis=-1)
    dots = np.clip(dots, -1.0, 1.0)
    return np.arccos(dots)  # 라디안



def pair_cost_matrix_joint_mid(
    A_q: np.ndarray, A_R: np.ndarray, A_p: np.ndarray, A_Q: np.ndarray,
    B_q: np.ndarray, B_R: np.ndarray, B_p: np.ndarray, B_Q: np.ndarray,
    lam_rot: float,
    tool_z: float,
) -> np.ndarray:
    """
    두 클러스터(A, B)의 모든 IK쌍 (i,j)에 대해
    q_mid = 0.5*(q_i + q_j), T_mid = FK(q_mid)
    cost(i,j) = d(T_i, T_mid) + d(T_mid, T_j)
    를 계산하여 (SA, SB) 행렬로 반환.
    """
    SA, SB = A_q.shape[0], B_q.shape[0]

    # 더 작은 쪽을 바깥루프로
    swap = False
    if SB < SA:
        # 스왑해서 바깥 루프를 최소화
        A_q, B_q = B_q, A_q
        A_R, B_R = B_R, A_R
        A_p, B_p = B_p, A_p
        A_Q, B_Q = B_Q, A_Q
        SA, SB = SB, SA
        swap = True

    cost = np.empty((SA, SB), dtype=np.float64)

    # 배치 크기 설정 (메모리 효율)
    batch_size = min(64, SA)
    
    # 배치 처리로 메모리 사용량 줄이기
    for batch_start in range(0, SA, batch_size):
        batch_end = min(batch_start + batch_size, SA)
        batch_range = slice(batch_start, batch_end)
        n_batch = batch_end - batch_start
        
        # 배치의 모든 i에 대해 j 전체를 한번에 계산
        # q_mid[i,j] = 0.5 * (q_i + q_j)
        A_q_batch = A_q[batch_range]  # (n_batch, D)
        q_mid_all = 0.5 * (A_q_batch[:, None, :] + B_q[None, :, :])  # (n_batch, SB, D)
        q_mid_flat = q_mid_all.reshape(-1, A_q.shape[1])  # (n_batch*SB, D)
        
        # FK 배치 계산
        Rm, pm = fk_batch(q_mid_flat, tool_z)
        Rm = Rm.reshape(n_batch, SB, 3, 3)
        pm = pm.reshape(n_batch, SB, 3)
        
        # 거리 계산 (벡터화)
        A_p_batch = A_p[batch_range]  # (n_batch, 3)
        A_R_batch = A_R[batch_range]  # (n_batch, 3, 3)
        
        # d(Ti, Tmid)
        d1_pos = np.linalg.norm(pm - A_p_batch[:, None, :], axis=2)  # (n_batch, SB)
        th1 = rot_angle_ignore_tool_yaw(
            A_R_batch[:, None, :, :],  # (n_batch, 1, 3, 3)
            Rm  # (n_batch, SB, 3, 3)
        )  # (n_batch, SB)
        
        # d(Tmid, Tj)
        d2_pos = np.linalg.norm(B_p[None, :, :] - pm, axis=2)  # (n_batch, SB)
        th2 = rot_angle_ignore_tool_yaw(Rm, B_R[None, :, :, :])  # (n_batch, SB)
        
        cost[batch_range, :] = d1_pos + lam_rot * th1 + d2_pos + lam_rot * th2

    if swap:
        cost = cost.T  # (원래 SA,SB)로 복원

    return cost


# =============================
# HDF5 로드 & 클러스터 구성 (변경 없음)
# =============================
def build_clusters_from_h5(
    h5_path: str,
    use_safe_only: bool,
    tool_z: float,
    group_prefix: str = "viewpoint_",
) -> Tuple[List[Dict], np.ndarray, List[int]]:
    """
    반환:
      clusters: len=M (빈 뷰포인트 제거 후), 각 항목:
        {
          "q": (S,DOF),
          "R": (S,3,3),
          "p": (S,3),
          "Q": (S,4),     # quat
          "target": (3,), # world_pose[:3,3]
        }
      target_coords: (M,3)  (각 뷰포인트 world_pose 위치)
      nonempty_map:  원본 viewpoint 인덱스 -> (축소 후) 인덱스 (빈 경우 -1)
    """
    clusters: List[Dict] = []
    target_coords_list = []
    nonempty_map: List[int] = []

    with h5py.File(h5_path, "r") as f:
        # viewpoint_* 그룹 나열 (소팅)
        keys = sorted([k for k in f.keys() if k.startswith(group_prefix)])

        for gi, gname in enumerate(keys):
            g = f[gname]
            # 필수: world_pose(4x4), all_ik_solutions(S,DOF)
            if "world_pose" not in g or "all_ik_solutions" not in g:
                nonempty_map.append(-1)
                continue

            world_pose = np.array(g["world_pose"], dtype=np.float64)
            target = world_pose[:3, 3].astype(np.float64)

            q_all = np.array(g["all_ik_solutions"], dtype=np.float64)  # (S, DOF)
            if q_all.ndim != 2 or q_all.shape[0] == 0:
                # IK 없음
                nonempty_map.append(-1)
                continue

            # 충돌 마스크 처리 (기본: 안전해만 사용)
            if use_safe_only and "collision_free_mask" in g:
                m = np.array(g["collision_free_mask"], dtype=bool).reshape(-1)
                if m.shape[0] == q_all.shape[0]:
                    q_all = q_all[m]
                    if q_all.shape[0] == 0:
                        nonempty_map.append(-1)
                        continue

            # 엔드포인트 FK 미리 계산
            R, p = fk_batch(q_all, tool_z)
            Q = rot_to_quat_batch(R)

            clusters.append({
                "q": q_all,    # (S,DOF)
                "R": R,        # (S,3,3)
                "p": p,        # (S,3)
                "Q": Q,        # (S,4)
                "target": target,  # (3,)
            })
            target_coords_list.append(target)
            nonempty_map.append(len(clusters) - 1)

    if len(clusters) == 0:
        raise RuntimeError("No non-empty viewpoints found in HDF5.")

    target_coords = np.stack(target_coords_list, axis=0)  # (M,3)
    return clusters, target_coords, nonempty_map


# =============================
# 이웃 그래프(KNN) (변경 없음)
# =============================
def build_neighbors_knn(target_coords: np.ndarray, k: int) -> List[List[int]]:
    M = target_coords.shape[0]
    nbrs = [[] for _ in range(M)]
    # 거리 행렬
    dif = target_coords[:, None, :] - target_coords[None, :, :]
    dist = np.linalg.norm(dif, axis=-1)
    np.fill_diagonal(dist, np.inf)
    for i in range(M):
        idx = np.argsort(dist[i])[:max(1, min(k, M-1))]
        nbrs[i] = idx.tolist()
    return nbrs

# =============================
# ★ [로직 복원] ★ 이웃 그래프 (Auto + MST 보강)
# =============================
def build_neighbors_auto(target_coords: np.ndarray) -> List[List[int]]:
    """
    '거리 점프 비율' + MST 보강 (이전 코드 방식)
    """
    M = target_coords.shape[0]
    if M == 0:
        return []
        
    dif = target_coords[:, None, :] - target_coords[None, :, :]
    dist = np.linalg.norm(dif, axis=-1)
    np.fill_diagonal(dist, np.inf)

    # 1) '거리 점프 비율' 기반 이웃
    nbrs = [[] for _ in range(M)]
    for i in range(M):
        order = np.argsort(dist[i])              # 가까운 순
        di = dist[i, order]
        m = np.isfinite(di).sum()
        if m == 0:
            nbrs[i] = []
            continue
        di = di[:m]; ord_use = order[:m]

        if m == 1:
            cand = [int(ord_use[0])]
        else:
            ratios = di[1:] / np.maximum(di[:-1], 1e-12)  # 인접 거리 비율
            j_star = int(np.argmax(ratios))               # 가장 큰 점프
            cut = max(0, j_star)                          # 최소 1개 보장
            cand = ord_use[:cut+1].tolist()

        if len(cand) == 0 and m > 0: # 비상 가드
            cand = [int(ord_use[0])]
        nbrs[i] = cand

    # 2) MST (Prim's)
    visited = np.zeros(M, dtype=bool)
    if M > 0:
        visited[0] = True
    edges = []
    for _ in range(M - 1):
        best = (np.inf, -1, -1)
        
        in_nodes = np.where(visited)[0]
        out_nodes = np.where(~visited)[0]
        if out_nodes.size == 0:
            break
            
        # 모든 (in, out) 엣지 중 최소 찾기
        D_sub = dist[in_nodes, :][:, out_nodes]
        if D_sub.size == 0:
            break # 분리된 그래프
            
        min_idx_flat = np.argmin(D_sub)
        min_idx_in, min_idx_out = np.unravel_index(min_idx_flat, D_sub.shape)
        u, v = in_nodes[min_idx_in], out_nodes[min_idx_out]
        w = D_sub[min_idx_in, min_idx_out]
        
        if w == np.inf:
            break # 연결 불가

        visited[v] = True
        edges.append((u, v))

    # 3) MST 간선 보강
    for u, v in edges:
        if v not in nbrs[u]:
            nbrs[u].append(v)
        if u not in nbrs[v]:
            nbrs[v].append(u)
            
    # 4) 비상 가드: 이웃이 비면 최근접 하나
    for i in range(M):
        if len(nbrs[i]) == 0 and M > 1:
            nbrs[i] = [int(np.argmin(dist[i]))]

    return nbrs


# =============================
# ★ [로직 복원] ★ 순서 생성 (실제 로봇 비용 기반 그리디)
# =============================
def build_visit_order_robot_cost(
    clusters: List[Dict],
    nbrs: List[List[int]],
    lam_rot: float,
    tool_z: float
) -> List[int]:
    """
    시작점: 원점(0,0,0)에서 가장 가까운 타깃.
    [로직 복원] 현재 노드의 이웃 중 아직 미방문이고 "실제 로봇 비용"이
    가장 적은 노드를 우선 방문, 없으면 전체 중 미방문 최소 비용 노드를 선택.
    """
    M = len(clusters)
    if M == 0:
        return []

    # 1) 비용 계산 헬퍼 (캐시 사용)
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

    # 2) 시작점 (좌표 기준)
    target_coords = np.stack([c["target"] for c in clusters], axis=0)
    start = int(np.argmin(np.linalg.norm(target_coords, axis=1)))
    
    order = [start]
    visited = np.zeros(M, dtype=bool)
    visited[start] = True
    cur = start

    # 3) 그리디 탐색 (로봇 비용 기준)
    for _ in range(M - 1):
        best_cost, best_j = np.inf, -1
        
        # 이웃 내에서 우선 탐색
        cand = [v for v in nbrs[cur] if not visited[v]]
        
        # 이웃이 다 방문됨 → 전체 미방문 중 최소
        if not cand:
            cand = [v for v in range(M) if not visited[v]]
        
        if not cand: # 방문할 곳이 없음
            break 
            
        for j in cand:
            c = pair_cost(cur, j) # ★ 실제 로봇 비용 사용
            if c < best_cost:
                best_cost, best_j = c, j
        
        if best_j == -1: # 연결할 노드가 없음 (그래프 분리)
            print(f"[WARN] TSP greedy search stuck at node {cur}. Graph may be disconnected.", file=sys.stderr)
            break
        
        visited[best_j] = True
        order.append(best_j)
        cur = best_j
        
    return order


# =============================
# DP로 IK 연쇄 최적화 (변경 없음)
# =============================
def choose_ik_given_order(
    clusters: List[Dict],
    order: List[int],
    lam_rot: float,
    tool_z: float,
) -> Tuple[Dict[int, int], float]:
    """
    고정된 순서 order(인덱스는 clusters 기준)에서
    각 지점의 IK를 선택해 총 비용을 최소화.
    반환:
      picked: {cluster_idx -> ik_idx}
      total_cost: 최소 총 비용
    """
    n = len(order)
    if n == 0:
        return {}, 0.0

    # 각 구간의 비용 행렬 미리 계산
    # C_k: (S_k, S_{k+1}) for k=0..n-2
    cost_mats: List[np.ndarray] = []
    sizes = []
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
        )  # (Sa, Sb)
        cost_mats.append(C)
        sizes.append((A_q.shape[0], B_q.shape[0]))

    # DP
    # dp[t][i] = t번째 노드(=order[t])에서 IK i를 선택했을 때의 최소 누적비용
    # back[t][i] = 이전 IK 인덱스
    Sa0 = clusters[order[0]]["q"].shape[0]
    dp = [np.full((Sa0,), 0.0, dtype=np.float64)]
    back = [np.full((Sa0,), -1, dtype=np.int32)]

    for t in range(n - 1):
        C = cost_mats[t]             # (Sa, Sb)
        Sa, Sb = C.shape
        prev = dp[-1]                # (Sa,)
        # 다음 단계 dp 계산: dp_next[j] = min_i dp[i] + C[i,j]
        # 이 연산은 (Sa, Sb) 브로드캐스팅으로 가능
        tmp = prev[:, None] + C      # (Sa, Sb)
        dp_next = np.min(tmp, axis=0)            # (Sb,)
        arg = np.argmin(tmp, axis=0).astype(np.int32)  # (Sb,)
        dp.append(dp_next)
        back.append(arg)

    # 마지막에서 최소 IK 선택
    last = dp[-1]
    j_star = int(np.argmin(last))
    total_cost = float(last[j_star])

    # 역추적
    picked_local = [None] * n
    picked_local[-1] = j_star
    for t in range(n - 2, -1, -1):
        i_star = int(back[t + 1][picked_local[t + 1]])
        picked_local[t] = i_star

    # 매핑: cluster_idx -> ik_idx
    picked = {}
    for t, cidx in enumerate(order):
        picked[cidx] = int(picked_local[t])

    return picked, total_cost


# =============================
# CSV 출력 (변경 없음)
# =============================
# def export_to_csv(
#     csv_path: str,
#     order: List[int],
#     picked: Dict[int, int],
#     clusters: List[Dict],
# ):
#     import csv
#     if not order:
#         print("[WARN] Cannot write CSV, order is empty.", file=sys.stderr)
#         return
        
#     with open(csv_path, "w", newline="") as f:
#         w = csv.writer(f)
#         # 헤더
#         dof = clusters[order[0]]["q"].shape[1]
#         q_headers = [f"q{i}" for i in range(dof)]
#         r_headers = [f"r{r}{c}" for r in range(3) for c in range(3)]
#         header = ["step", "target_idx(local)", "ik_idx"] + q_headers + ["px", "py", "pz"] + r_headers
#         w.writerow(header)

#         for step, cidx in enumerate(order):
#             if cidx not in picked:
#                 print(f"[WARN] Skipping CSV row: target_idx {cidx} not in picked map.", file=sys.stderr)
#                 continue
#             cl = clusters[cidx]
#             ik = picked[cidx]
#             q = cl["q"][ik]
#             p = cl["p"][ik]
#             R = cl["R"][ik]
#             row = [step, cidx, ik] + q.tolist() + p.tolist() + R.reshape(-1).tolist()
#             w.writerow(row)

def export_to_csv(
    csv_path: str,
    order: List[int],
    picked: Dict[int, int],
    clusters: List[Dict],
):
    """
    joint_trajectory_dp_5000_repaired.csv 스타일로 내보내기

    Columns:
      time,
      ur20-shoulder_pan_joint,
      ur20-shoulder_lift_joint,
      ur20-elbow_joint,
      ur20-wrist_1_joint,
      ur20-wrist_2_joint,
      ur20-wrist_3_joint,
      target-POS_X, target-POS_Y, target-POS_Z,
      target-ROT_X, target-ROT_Y, target-ROT_Z, target-ROT_W
    """
    import csv, sys

    if not order:
        print("[WARN] Cannot write CSV: order is empty.", file=sys.stderr)
        return

    headers = [
        "time",
        "ur20-shoulder_pan_joint",
        "ur20-shoulder_lift_joint",
        "ur20-elbow_joint",
        "ur20-wrist_1_joint",
        "ur20-wrist_2_joint",
        "ur20-wrist_3_joint",
        "target-POS_X", "target-POS_Y", "target-POS_Z",
        "target-ROT_X", "target-ROT_Y", "target-ROT_Z", "target-ROT_W",
    ]

    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(headers)

        for step, cidx in enumerate(order):
            if cidx not in picked:
                print(f"[WARN] Skipping row: target_idx {cidx} not in picked.", file=sys.stderr)
                continue

            cl = clusters[cidx]
            ik = picked[cidx]

            # 조인트 6개만 기록(UR20 기준), 부족하면 0으로 패딩
            q = cl["q"][ik]
            if q.shape[0] >= 6:
                q6 = q[:6]
            else:
                q6 = list(q) + [0.0] * (6 - len(q))

            # 타깃 위치/자세: world_pose 기반이 있으면 사용, 없으면 FK 결과로 대체
            # 위치
            pos = cl["target"] if "target" in cl else cl["p"][ik]
            # 자세(쿼터니언): (w,x,y,z) → CSV(X,Y,Z,W)
            Qwxyz = cl["target_Q"] if "target_Q" in cl else cl["Q"][ik]
            rot_x, rot_y, rot_z, rot_w = Qwxyz[1], Qwxyz[2], Qwxyz[3], Qwxyz[0]

            row = [
                float(step),                  # time: 스텝 번호를 그대로 사용(1.0 간격)
                float(q6[0]), float(q6[1]), float(q6[2]),
                float(q6[3]), float(q6[4]), float(q6[5]),
                float(pos[0]), float(pos[1]), float(pos[2]),
                float(rot_x), float(rot_y), float(rot_z), float(rot_w),
            ]
            w.writerow(row)



# =============================
# 메인 (★ 로직 호출 수정 ★)
# =============================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--h5", required=True, help="HDF5 file with viewpoint_* groups")
    ap.add_argument("--lam_rot", type=float, default=0.7, help="rotation weight in distance")
    ap.add_argument("--tool_len", type=float, default=0.0, help="tool TCP offset along +Z (m)")
    ap.add_argument("--allow_unsafe", action="store_true", help="use all IKs (ignore collision_free_mask)")
    ap.add_argument("--knn", type=int, default=0, help="K for KNN neighbor graph; 0=auto(jump-ratio)+MST")
    ap.add_argument("--csv_out", type=str, default="", help="output CSV path")
    args = ap.parse_args()

    use_safe_only = (not args.allow_unsafe)
    tool_z = float(args.tool_len)
    start_time = time.time()
    print(f"[INFO] Loading H5: {args.h5}")
    clusters, target_coords, nonempty_map = build_clusters_from_h5(
        args.h5, use_safe_only=use_safe_only, tool_z=tool_z
    )
    M = len(clusters)
    print(f"[INFO] Non-empty viewpoints: {M}")

    # ★ [로직 복원] ★ 이웃 그래프
    if args.knn and args.knn > 0:
        nbrs = build_neighbors_knn(target_coords, k=args.knn)
        print(f"[INFO] Neighbor graph: KNN K={args.knn} (connectivity not guaranteed)")
    else:
        # ★ 'build_neighbors_auto_mst' 대신 'build_neighbors_auto' 호출
        nbrs = build_neighbors_auto(target_coords)
        print(f"[INFO] Neighbor graph: AUTO (Jump-Ratio) + MST")

    # ★ [로직 복원] ★ 순서 생성 (실제 로봇 비용 기반)
    print("[INFO] Visit order: Greedy on ROBOT COST (pair_min_ee_cost)")
    # ★ 'build_visit_order' 대신 'build_visit_order_robot_cost' 호출
    order = build_visit_order_robot_cost(
        clusters=clusters,
        nbrs=nbrs,
        lam_rot=args.lam_rot,
        tool_z=tool_z,
    )
    print(f"[INFO] Visit order (local indices 0..{M-1}): {order}")

    # DP로 IK 선택 (변경 없음)
    print("[INFO] Optimizing IK chain using Dynamic Programming...")
    picked, total_cost = choose_ik_given_order(
        clusters=clusters,
        order=order,
        lam_rot=args.lam_rot,
        tool_z=tool_z,
    )
    print(f"[RESULT] Total cost (EE length approx, mid=joint-lerp 1pt): {total_cost:.6f}")
    print(f"Total time: {time.time()-start_time}")

    # CSV 저장(옵션) (변경 없음)
    if args.csv_out:
        export_to_csv(args.csv_out, order, picked, clusters)
        print(f"[INFO] Saved CSV to: {args.csv_out}")

    # 원본 인덱스 매핑 정보(참고 출력)
    # 여기서는 clusters가 이미 빈 지점을 제거한 상태의 로컬 인덱스임.
    # 필요하면 nonempty_map을 활용해 원본 viewpoint_* 인덱스와 매칭 가능.
    # 예: order_original = [nonempty_map[i] for i in order] (nonempty_map이 리스트가 아닌 경우 수정 필요)
    return 0


if __name__ == "__main__":
    sys.exit(main())