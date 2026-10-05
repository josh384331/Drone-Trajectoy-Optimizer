#!/usr/bin/env python3
"""
trajectory_optimizer.py

Full integrated A* + CasADi trajectory optimizer with environment I/O and visualization.
Requires: numpy, casadi, plotly
"""

import json
import argparse
import numpy as np
import heapq
import casadi as ca
import plotly.graph_objects as go
from datetime import datetime

# -----------------------------
# Environment helpers and I/O
# -----------------------------
def default_environment():
    """Return a default environment dict (grid units are voxels)."""
    env = {
        "resolution": 1.0,
        "size": [40, 40, 20],  # nx, ny, nz (voxel counts)
        "obstacles": [
            {"type": "cylinder", "center": [20, 20], "radius": 5, "z_min": 0, "z_max": 10},
            {"type": "box", "min": [12, 28, 0], "max": [22, 36, 8]},
            {"type": "box", "min": [10, 10, 0], "max": [14, 30, 6]}
        ],
        "risk_fields": [
            {"type": "gaussian", "center": [30.0, 20.0], "sigma": 6.0, "altitude_decay": 0.1},
            {"type": "gaussian", "center": [10.0, 30.0], "sigma": 6.0, "altitude_decay": 0.1}
        ]
    }
    return env

def load_environment(path):
    with open(path, 'r') as f:
        env = json.load(f)
    return env

def build_occ_from_environment(env):
    nx, ny, nz = env["size"]
    occ = np.zeros((nz, ny, nx), dtype=bool)
    for obj in env.get("obstacles", []):
        typ = obj.get("type", "")
        if typ == "box":
            xmin, ymin, zmin = obj["min"]
            xmax, ymax, zmax = obj["max"]
            xmin = int(max(0, xmin)); ymin = int(max(0, ymin)); zmin = int(max(0, zmin))
            xmax = int(min(nx, xmax)); ymax = int(min(ny, ymax)); zmax = int(min(nz, zmax))
            occ[zmin:zmax, ymin:ymax, xmin:xmax] = True
        elif typ == "cylinder":
            cx, cy = obj["center"]
            R = obj["radius"]
            zmin = int(max(0, obj.get("z_min", 0)))
            zmax = int(min(nz, obj.get("z_max", nz)))
            for z in range(zmin, zmax):
                for y in range(ny):
                    for x in range(nx):
                        if (x - cx) ** 2 + (y - cy) ** 2 <= R * R:
                            occ[z, y, x] = True
        elif typ == "sphere":
            cx, cy, cz = obj["center"]
            R = obj["radius"]
            for z in range(nz):
                for y in range(ny):
                    for x in range(nx):
                        if (x - cx) ** 2 + (y - cy) ** 2 + (z - cz) ** 2 <= R * R:
                            occ[z, y, x] = True
    return occ

def build_risk_from_environment(env):
    nx, ny, nz = env["size"]
    res = env.get("resolution", 1.0)
    risk = np.zeros((nz, ny, nx), dtype=float)
    xs = np.arange(nx) * res
    ys = np.arange(ny) * res
    zs = np.arange(nz) * res
    X, Y = np.meshgrid(xs, ys, indexing='xy')
    for rf in env.get("risk_fields", []):
        if rf.get("type") == "gaussian":
            cx, cy = rf["center"]
            sigma = rf.get("sigma", max(nx, ny) * res / 6.0)
            beta = rf.get("altitude_decay", 0.12)
            base = np.exp(-((X - cx) ** 2 + (Y - cy) ** 2) / (2 * sigma * sigma))
            fz = np.exp(-beta * zs)
            for k in range(nz):
                risk[k, :, :] += base * (1.0 + fz[k] * 2.0)
    # normalize
    risk /= (risk.max() + 1e-12)
    return risk

# -----------------------------
# 3D A* (grid-based)
# -----------------------------
def neighbors_3d():
    nbrs = []
    for dz in [-1, 0, 1]:
        for dy in [-1, 0, 1]:
            for dx in [-1, 0, 1]:
                if dx == 0 and dy == 0 and dz == 0:
                    continue
                move_cost = np.sqrt(dx * dx + dy * dy + dz * dz)
                nbrs.append((dx, dy, dz, move_cost))
    return nbrs

def idx_from_xyz(x, y, z, nx, ny, nz):
    return int(x + nx * (y + ny * z))

def xyz_from_idx(idx, nx, ny, nz):
    z = idx // (nx * ny)
    rem = idx % (nx * ny)
    y = rem // nx
    x = rem % nx
    return int(x), int(y), int(z)

def astar_3d(start, goal, occ, risk=None, w_risk=5.0, base_cost=1.0):
    nz, ny, nx = occ.shape
    nbrs = neighbors_3d()
    start_idx = idx_from_xyz(*start, nx, ny, nz)
    goal_idx = idx_from_xyz(*goal, nx, ny, nz)
    g = np.full(nx * ny * nz, np.inf, dtype=np.float32)
    parent = np.full(nx * ny * nz, -1, dtype=np.int32)
    def heuristic(x, y, z):
        gx, gy, gz = goal
        return base_cost * np.sqrt((x - gx) ** 2 + (y - gy) ** 2 + (z - gz) ** 2)
    open_heap = []
    g[start_idx] = 0.0
    heapq.heappush(open_heap, (heuristic(*start), start_idx))
    in_open = np.zeros(nx * ny * nz, dtype=bool)
    in_open[start_idx] = True
    while open_heap:
        f_curr, curr_idx = heapq.heappop(open_heap)
        in_open[curr_idx] = False
        if curr_idx == goal_idx:
            path = []
            idx = curr_idx
            while idx != -1:
                x, y, z = xyz_from_idx(idx, nx, ny, nz)
                path.append((x, y, z))
                idx = parent[idx]
            path.reverse()
            return path
        cx, cy, cz = xyz_from_idx(curr_idx, nx, ny, nz)
        for dx, dy, dz, move_cost in nbrs:
            nx_ = cx + dx
            ny_ = cy + dy
            nz_ = cz + dz
            if nx_ < 0 or nx_ >= nx or ny_ < 0 or ny_ >= ny or nz_ < 0 or nz_ >= nz:
                continue
            if occ[nz_, ny_, nx_]:
                continue
            n_idx = idx_from_xyz(nx_, ny_, nz_, nx, ny, nz)
            step_cost = base_cost * move_cost
            if risk is not None:
                step_cost += w_risk * risk[nz_, ny_, nx_]
            tentative_g = g[curr_idx] + step_cost
            if tentative_g < g[n_idx]:
                g[n_idx] = tentative_g
                parent[n_idx] = curr_idx
                f_n = tentative_g + heuristic(nx_, ny_, nz_)
                if not in_open[n_idx]:
                    heapq.heappush(open_heap, (f_n, n_idx))
                    in_open[n_idx] = True
    return None

# -----------------------------
# Utilities: resample path and trilinear interpolation (numpy)
# -----------------------------
def resample_path(path_voxels, num_nodes):
    pts = np.array(path_voxels, dtype=float)
    diffs = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    cum = np.concatenate(([0.0], np.cumsum(diffs)))
    if cum[-1] == 0:
        return np.repeat(pts[:1], num_nodes, axis=0)
    t = np.linspace(0, cum[-1], num_nodes)
    resampled = np.zeros((num_nodes, 3), dtype=float)
    for i in range(3):
        resampled[:, i] = np.interp(t, cum, pts[:, i])
    return resampled

def trilinear_interpolate(field, p):
    nz, ny, nx = field.shape
    x, y, z = p
    x = np.clip(x, 0, nx - 1 - 1e-6)
    y = np.clip(y, 0, ny - 1 - 1e-6)
    z = np.clip(z, 0, nz - 1 - 1e-6)
    x0 = int(np.floor(x)); x1 = x0 + 1
    y0 = int(np.floor(y)); y1 = y0 + 1
    z0 = int(np.floor(z)); z1 = z0 + 1
    xd = x - x0; yd = y - y0; zd = z - z0
    c000 = field[z0, y0, x0]; c100 = field[z0, y0, x1]
    c010 = field[z0, y1, x0]; c110 = field[z0, y1, x1]
    c001 = field[z1, y0, x0]; c101 = field[z1, y0, x1]
    c011 = field[z1, y1, x0]; c111 = field[z1, y1, x1]
    c00 = c000 * (1 - xd) + c100 * xd
    c01 = c001 * (1 - xd) + c101 * xd
    c10 = c010 * (1 - xd) + c110 * xd
    c11 = c011 * (1 - xd) + c111 * xd
    c0 = c00 * (1 - yd) + c10 * yd
    c1 = c01 * (1 - yd) + c11 * yd
    c = c0 * (1 - zd) + c1 * zd
    return float(c)

# -----------------------------
# Obstacle constraint generator (CasADi expressions)
# -----------------------------
def obstacle_constraints_from_env(env, p, margin=1.0):
    """
    Return list of CasADi expressions g_i(p) such that g_i >= 0 enforces keep-out.
    Supports box, cylinder, sphere.
    margin: safety margin in voxels
    """
    g_list = []
    for obj in env.get("obstacles", []):
        typ = obj.get("type", "")
        if typ == "cylinder":
            cx, cy = obj["center"]
            R = float(obj["radius"]) + margin
            # outside cylinder: (x-cx)^2 + (y-cy)^2 - R^2 >= 0
            g_list.append((p[0] - cx) ** 2 + (p[1] - cy) ** 2 - (R ** 2))
        elif typ == "sphere":
            cx, cy, cz = obj["center"]
            R = float(obj["radius"]) + margin
            g_list.append((p[0] - cx) ** 2 + (p[1] - cy) ** 2 + (p[2] - cz) ** 2 - (R ** 2))
        elif typ == "box":
            xmin, ymin, zmin = obj["min"]
            xmax, ymax, zmax = obj["max"]
            # distance from point to box (0 if inside). We require distance >= margin.
            # dx = max(xmin - x, 0, x - xmax)
            dx = ca.fmax(ca.fmax(xmin - p[0], 0.0), p[0] - xmax)
            dy = ca.fmax(ca.fmax(ymin - p[1], 0.0), p[1] - ymax)
            dz = ca.fmax(ca.fmax(zmin - p[2], 0.0), p[2] - zmax)
            dist_sq = dx * dx + dy * dy + dz * dz
            g_list.append(dist_sq - (margin ** 2))
        # other shapes can be added here
    return g_list

# -----------------------------
# CasADi NLP builder (analytic risk + obstacle constraints)
# -----------------------------
def build_casadi_nlp(start, goal, N, env,
                     w_acc=1.0, w_len=0.1, w_risk=5.0, margin=1.0):
    """
    Build CasADi NLP. start, goal are arrays in voxel coords (x,y,z).
    env: environment dict (used for analytic risk and obstacle constraints).
    Returns: nlp dict, lbx, ubx, lbg, ubg
    """
    nx, ny, nz = env["size"]
    res = env.get("resolution", 1.0)

    n_interior = N - 2
    X = ca.SX.sym('X', n_interior * 3)

    def node_sym(k):
        if k == 0:
            return ca.DM(start)
        elif k == N - 1:
            return ca.DM(goal)
        else:
            i = (k - 1) * 3
            return X[i:i + 3]

    # analytic risk parameters (reuse same structure as build_risk_from_environment)
    # We'll sum Gaussians from env risk_fields
    risk_fields = env.get("risk_fields", [])
    def risk_analytic(p):
        # p in voxel coords; convert to meters by res
        x = p[0] * res
        y = p[1] * res
        z = p[2] * res
        rsum = 0
        for rf in risk_fields:
            if rf.get("type") == "gaussian":
                cx, cy = rf["center"]
                sigma = rf.get("sigma", max(nx, ny) * res / 6.0)
                beta = rf.get("altitude_decay", 0.12)
                dx1 = x - cx
                dy1 = y - cy
                g = ca.exp(-(dx1 * dx1 + dy1 * dy1) / (2 * sigma * sigma))
                fz = ca.exp(-beta * z)
                rsum = rsum + g * (1.0 + 2.0 * fz)
        # optional normalization omitted (scales are tunable via w_risk)
        return rsum

    # Build objective
    J = 0
    for k in range(1, N - 1):
        x_km1 = node_sym(k - 1)
        x_k = node_sym(k)
        x_kp1 = node_sym(k + 1)
        acc = x_kp1 - 2 * x_k + x_km1
        J += w_acc * ca.sumsqr(acc)
    for k in range(N - 1):
        p0 = node_sym(k)
        p1 = node_sym(k + 1)
        seg = p1 - p0
        J += w_len * ca.norm_2(seg)
    for k in range(N):
        p = node_sym(k)
        # clamp inside grid
        px = ca.fmax(0.0, ca.fmin(p[0], nx - 1 - 1e-6))
        py = ca.fmax(0.0, ca.fmin(p[1], ny - 1 - 1e-6))
        pz = ca.fmax(0.0, ca.fmin(p[2], nz - 1 - 1e-6))
        J += w_risk * risk_analytic(ca.vertcat(px, py, pz))

    # Obstacle constraints
    g_exprs = []
    for k in range(N):
        p = node_sym(k)
        g_list = obstacle_constraints_from_env(env, p, margin=margin)
        g_exprs.extend(g_list)

    if len(g_exprs) > 0:
        g = ca.vertcat(*g_exprs)
        lbg = [0.0] * g.size1()
        ubg = [ca.inf] * g.size1()
    else:
        g = ca.vertcat()
        lbg = []
        ubg = []

    # bounds for interior nodes
    lbx = []
    ubx = []
    for i in range(n_interior):
        lbx += [0.0, 0.0, 0.0]
        ubx += [nx - 1 - 1e-6, ny - 1 - 1e-6, nz - 1 - 1e-6]

    nlp = {'x': X, 'f': J, 'g': g}
    return nlp, lbx, ubx, lbg, ubg

# -----------------------------
# Plotting with Plotly (A* + optimized)
# -----------------------------
def visualize_3d(occ, risk, opt_nodes, coarse_nodes=None, res=1.0, title="Trajectory Optimization"):
    nz, ny, nx = occ.shape
    xs = np.arange(nx) * res
    ys = np.arange(ny) * res
    zs = np.arange(nz) * res

    data = []
    occ_indices = np.argwhere(occ)
    if occ_indices.size > 0:
        ox = occ_indices[:, 2] * res
        oy = occ_indices[:, 1] * res
        oz = occ_indices[:, 0] * res
        data.append(go.Scatter3d(x=ox, y=oy, z=oz, mode='markers',
                                 marker=dict(size=3, color='black'), name='Obstacles'))

    # Risk volume (sparse sampling to keep plot light)
    Xg = np.repeat(xs[np.newaxis, np.newaxis, :], ny, axis=1).repeat(nz, axis=0).flatten()
    Yg = np.repeat(ys[np.newaxis, :, np.newaxis], nx, axis=2).repeat(nz, axis=0).flatten()
    Zg = np.repeat(zs[:, np.newaxis, np.newaxis], ny, axis=1).repeat(nx, axis=2).flatten()
    data.append(go.Volume(x=Xg, y=Yg, z=Zg, value=risk.flatten(),
                          isomin=0.05, isomax=1.0, opacity=0.12, surface_count=12,
                          colorscale='YlOrRd', name='Risk'))

    # A* coarse path (green dashed)
    if coarse_nodes is not None:
        px = coarse_nodes[:, 0] * res
        py = coarse_nodes[:, 1] * res
        pz = coarse_nodes[:, 2] * res
        data.append(go.Scatter3d(x=px, y=py, z=pz, mode='lines+markers',
                                 line=dict(color='green', width=3, dash='dash'),
                                 marker=dict(size=3, color='green'),
                                 name='A* Path'))

    # Optimized path (blue)
    if opt_nodes is not None:
        px = opt_nodes[:, 0] * res
        py = opt_nodes[:, 1] * res
        pz = opt_nodes[:, 2] * res
        data.append(go.Scatter3d(x=px, y=py, z=pz, mode='lines+markers',
                                 line=dict(color='blue', width=5),
                                 marker=dict(size=4, color='blue'),
                                 name='Optimized Path'))

    fig = go.Figure(data=data)
    fig.update_layout(scene=dict(xaxis_title='X (m)', yaxis_title='Y (m)', zaxis_title='Z (m)', aspectmode='data'),
                      title=title)
    fig.show()
    uniqueDateTime = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    fig.write_html(f"optimized_trajectory_{uniqueDateTime}.html")

# -----------------------------
# Demo: run A*, resample, optimize, visualize
# -----------------------------
def demo_optimize(env=None, start=(5.0,5.0,0.0), goal=(35.0,35.0,0.0), N=30):
    if env is None:
        env = default_environment()
    res = env.get("resolution", 1.0)
    nx, ny, nz = env["size"]

    occ = build_occ_from_environment(env)
    risk = build_risk_from_environment(env)

    start_vox = (int(start[0]), int(start[1]), int(start[2]))
    goal_vox = (int(goal[0]), int(goal[1]), int(goal[2]))
    coarse_path = astar_3d(start_vox, goal_vox, occ, risk, w_risk=8.0, base_cost=1.0)
    if coarse_path is None:
        print("No coarse path found.")
        return
    print("Coarse path length (voxels):", len(coarse_path))

    X0 = resample_path(coarse_path, N)  # initial guess
    x0_flat = X0[1:-1].ravel()

    # weights
    w_acc = 3.0
    w_len = 0.01
    w_risk = 0.2
    margin = 0.5  # safety margin in voxels

    nlp, lbx, ubx, lbg, ubg = build_casadi_nlp(
        start=np.array(start, dtype=float),
        goal=np.array(goal, dtype=float),
        N=N,
        env=env,
        w_acc=w_acc, w_len=w_len, w_risk=w_risk,
        margin=margin
    )

    opts = {'ipopt.print_level': 3, 'print_time': False, 'ipopt.max_iter': 400, 'ipopt.tol': 1e-6}
    solver = ca.nlpsol('solver', 'ipopt', nlp, opts)

    print("Starting CasADi/IPOPT optimization...")
    sol = solver(x0=x0_flat, lbx=lbx, ubx=ubx, lbg=lbg, ubg=ubg)
    x_opt = sol['x'].full().ravel()

    X_opt = np.zeros((N, 3), dtype=float)
    X_opt[0] = start
    X_opt[-1] = goal
    X_opt[1:-1] = x_opt.reshape((N - 2, 3))

    segs_before = np.linalg.norm(np.diff(X0, axis=0), axis=1)
    segs_after = np.linalg.norm(np.diff(X_opt, axis=0), axis=1)
    len_before = segs_before.sum()
    len_after = segs_after.sum()
    risk_before = sum(trilinear_interpolate(risk, p) for p in X0)
    risk_after = sum(trilinear_interpolate(risk, p) for p in X_opt)
    print(f"Length before: {len_before:.3f}, after: {len_after:.3f}")
    print(f"Risk before: {risk_before:.3f}, after: {risk_after:.3f}")

    visualize_3d(occ, risk, X_opt, coarse_nodes=X0, res=res,
                 title="Optimized 3D Trajectory (CasADi with hard obstacles)")

# -----------------------------
# CLI
# -----------------------------
def main():
    parser = argparse.ArgumentParser(description="3D Trajectory Optimizer (A* + CasADi)")
    parser.add_argument("--env", type=str, default=None, help="Path to environment JSON file")
    parser.add_argument("--start", type=float, nargs=3, default=[5.0,5.0,0.0], help="Start voxel coords x y z")
    parser.add_argument("--goal", type=float, nargs=3, default=[35.0,35.0,0.0], help="Goal voxel coords x y z")
    parser.add_argument("--nodes", type=int, default=30, help="Number of trajectory nodes")
    args = parser.parse_args()

    env = None
    if args.env:
        env = load_environment(args.env)
    demo_optimize(env=env, start=tuple(args.start), goal=tuple(args.goal), N=args.nodes)

if __name__ == "__main__":
    main()
