# trajectory_module.py
"""
Trajectory optimizer module (A* + CasADi) with SDF-based hard keepout constraints.
Importable API:
  - default_environment()
  - build_environment(env)
  - compute_sdf(occ)
  - optimize_trajectory(env, start, goal, N, **kwargs)
  - visualize_3d(occ, risk, opt_nodes, coarse_nodes=None, res=1.0, title=...)
"""

import numpy as np
import heapq
import casadi as ca
import plotly.graph_objects as go
from datetime import datetime

# Try to import fast EDT
try:
    from scipy.ndimage import distance_transform_edt as edt
    _HAS_EDT = True
except Exception:
    edt = None
    _HAS_EDT = False

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
            {"type": "gaussian", "center": [30.0, 20.0], "sigma": 6.0, "altitude_decay": 0.12},
            {"type": "gaussian", "center": [10.0, 30.0], "sigma": 6.0, "altitude_decay": 0.12}
        ]
    }
    return env

def build_occ_from_environment(env):
    """Build occupancy grid (nz, ny, nx) from env dict."""
    nx, ny, nz = env["size"]
    occ = np.zeros((nz, ny, nx), dtype=bool)
    for obj in env.get("obstacles", []):
        typ = obj.get("type", "")
        if typ == "box":
            xmin, ymin, zmin = obj["min"]
            xmax, ymax, zmax = obj["max"]
            xmin = int(max(0, xmin)); ymin = int(max(0, ymin)); zmin = int(max(0, zmin))
            xmax = int(min(nx, xmax)); ymax = int(min(ny, ymax)); zmax = int(min(nz, zmax))
            if xmax > xmin and ymax > ymin and zmax > zmin:
                occ[zmin:zmax, ymin:ymax, xmin:xmax] = True
        elif typ == "cylinder":
            cx, cy = obj["center"]
            R = obj["radius"]
            zmin = int(max(0, obj.get("z_min", 0)))
            zmax = int(min(nz, obj.get("z_max", nz)))
            # vectorized fill per z-slice
            xs = np.arange(nx); ys = np.arange(ny)
            Xg, Yg = np.meshgrid(xs, ys, indexing='xy')
            mask2d = (Xg - cx) ** 2 + (Yg - cy) ** 2 <= R * R
            for z in range(zmin, zmax):
                occ[z, :, :] |= mask2d
        elif typ == "sphere":
            cx, cy, cz = obj["center"]
            R = obj["radius"]
            xs = np.arange(nx); ys = np.arange(ny); zs = np.arange(nz)
            Xg, Yg, Zg = np.meshgrid(xs, ys, zs, indexing='xy')
            # Note: meshgrid with 3D can be memory heavy; do slice-wise
            for z in range(nz):
                zz = z
                mask2d = (np.arange(nx)[None, :] - cx) ** 2 + (np.arange(ny)[:, None] - cy) ** 2 + (zz - cz) ** 2 <= R * R
                occ[z, :, :] |= mask2d
    return occ

def build_risk_from_environment(env):
    """Build analytic risk field sampled on grid (nz, ny, nx)."""
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
    risk /= (risk.max() + 1e-12)
    return risk

def build_environment(env):
    """Convenience: build occ, risk, res from env dict."""
    occ = build_occ_from_environment(env)
    risk = build_risk_from_environment(env)
    res = env.get("resolution", 1.0)
    return occ, risk, res

# -----------------------------
# Fast SDF builder
# -----------------------------
def compute_sdf(occ):
    """
    Compute signed distance field (distance to nearest obstacle surface).
    Returns SDF in voxel units (distance in voxels). Occupied voxels have distance 0.
    Uses scipy.ndimage.distance_transform_edt if available (fast).
    """
    if _HAS_EDT:
        # distance from free voxels to nearest occupied voxel
        # edt expects True for non-zero; we want distance to obstacles, so invert
        free = ~occ
        dist = edt(free)  # distance in voxels
        return dist.astype(float)
    else:
        # fallback: vectorized approximate EDT using broadcasting (O(n^2) worst-case)
        nz, ny, nx = occ.shape
        occ_pts = np.argwhere(occ)
        if occ_pts.size == 0:
            # no obstacles -> large distances
            return np.full_like(occ, fill_value=max(nx, ny, nz), dtype=float)
        # compute distance to nearest occupied point for each voxel
        # vectorized by flattening grid
        xs = np.arange(nx); ys = np.arange(ny); zs = np.arange(nz)
        Xg, Yg, Zg = np.meshgrid(xs, ys, zs, indexing='xy')
        pts = np.stack([Xg.ravel(), Yg.ravel(), Zg.ravel()], axis=1)
        # compute squared distances to all occ points in chunks to avoid memory blowup
        sdf = np.empty(pts.shape[0], dtype=float)
        occ_pts_arr = np.array(occ_pts, dtype=float)
        for i in range(0, pts.shape[0], 10000):
            chunk = pts[i:i+10000]
            # compute min distance to occ_pts
            d2 = np.sum((chunk[:, None, :] - occ_pts_arr[None, :, :]) ** 2, axis=2)
            sdf[i:i+chunk.shape[0]] = np.sqrt(np.min(d2, axis=1))
        sdf = sdf.reshape((nz, ny, nx))
        return sdf

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
# CasADi helper: trilinear interpolation using DM lookup (safe across versions)
# -----------------------------
def casadi_trilinear_dm(R_np, p, nx, ny, nz):
    """
    Build a CasADi expression for trilinear interpolation of a 3D numpy array at a symbolic point p.

    This must NOT use ca.DM(R_np.tolist()) because the SDF field is rank-3:
      R_np.shape == (nz, ny, nx)
    and CasADi DM only accepts 1D/2D numeric arrays, not a nested [[[float]]] list.

    Instead, use CasADi's interpolant() with x/y/z grids and a flattened value array.
    """
    R_np = np.asarray(R_np, dtype=float)
    if R_np.shape != (nz, ny, nx):
        raise ValueError(f"SDF array shape mismatch: expected {(nz, ny, nx)}, got {R_np.shape}")

    # Transpose to (x, y, z) ordering so the interpolant matches the axes used by the voxel grid.
    values = np.transpose(R_np, (2, 1, 0)).copy()
    values_flat = values.flatten(order='F').tolist()

    x_grid = np.arange(nx, dtype=float)
    y_grid = np.arange(ny, dtype=float)
    z_grid = np.arange(nz, dtype=float)

    # CasADi accepts the grid as a list of 1D arrays and the values as a flat list in Fortran order.
    interp = ca.interpolant('sdf_interp', 'linear', [x_grid, y_grid, z_grid], values_flat, {})

    # Clamp to the valid voxel domain so we never evaluate outside the array.
    x = ca.fmax(0.0, ca.fmin(p[0], nx - 1 - 1e-6))
    y = ca.fmax(0.0, ca.fmin(p[1], ny - 1 - 1e-6))
    z = ca.fmax(0.0, ca.fmin(p[2], nz - 1 - 1e-6))

    # CasADi interpolant expects a single 3-vector argument, not three scalar arguments.
    return interp(ca.vertcat(x, y, z))

# -----------------------------
# Obstacle constraints generator (SDF-based)
# -----------------------------
def sdf_constraints_from_sdf(sdf_np, p, nx, ny, nz, margin=1.0):
    """
    Return CasADi expression g(p) such that g >= 0 enforces sdf(p) - margin >= 0.
    Uses casadi_trilinear_dm to evaluate sdf at p.
    """
    sdf_expr = casadi_trilinear_dm(sdf_np, p, nx, ny, nz)
    return sdf_expr - margin

# -----------------------------
# CasADi NLP builder (analytic risk + SDF hard constraints)
# -----------------------------
def build_casadi_nlp_with_sdf(start, goal, N, env, sdf_np,
                              w_acc=1.0, w_len=0.1, w_risk=5.0, margin=1.0):
    """
    Build CasADi NLP using analytic risk and SDF hard constraints.
    start, goal: arrays in voxel coords (x,y,z)
    sdf_np: numpy array (nz, ny, nx) of distances in voxels
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

    # analytic risk fields
    risk_fields = env.get("risk_fields", [])
    def risk_analytic(p):
        x = p[0] * res; y = p[1] * res; z = p[2] * res
        rsum = 0
        for rf in risk_fields:
            if rf.get("type") == "gaussian":
                cx, cy = rf["center"]
                sigma = rf.get("sigma", max(nx, ny) * res / 6.0)
                beta = rf.get("altitude_decay", 0.12)
                dx1 = x - cx; dy1 = y - cy
                g = ca.exp(-(dx1 * dx1 + dy1 * dy1) / (2 * sigma * sigma))
                fz = ca.exp(-beta * z)
                rsum = rsum + g * (1.0 + 2.0 * fz)
        return rsum

    # objective
    J = 0
    for k in range(1, N - 1):
        x_km1 = node_sym(k - 1); x_k = node_sym(k); x_kp1 = node_sym(k + 1)
        acc = x_kp1 - 2 * x_k + x_km1
        J += w_acc * ca.sumsqr(acc)
    for k in range(N - 1):
        p0 = node_sym(k); p1 = node_sym(k + 1)
        J += w_len * ca.norm_2(p1 - p0)
    for k in range(N):
        p = node_sym(k)
        px = ca.fmax(0.0, ca.fmin(p[0], nx - 1 - 1e-6))
        py = ca.fmax(0.0, ca.fmin(p[1], ny - 1 - 1e-6))
        pz = ca.fmax(0.0, ca.fmin(p[2], nz - 1 - 1e-6))
        J += w_risk * risk_analytic(ca.vertcat(px, py, pz))

    # SDF constraints (g >= 0)
    g_list = []
    for k in range(N):
        p = node_sym(k)
        g_expr = sdf_constraints_from_sdf(sdf_np, p, nx, ny, nz, margin=margin)
        g_list.append(g_expr)
    g = ca.vertcat(*g_list)
    lbg = [0.0] * g.size1()
    ubg = [ca.inf] * g.size1()

    # bounds
    lbx = []; ubx = []
    for i in range(n_interior):
        lbx += [0.0, 0.0, 0.0]
        ubx += [nx - 1 - 1e-6, ny - 1 - 1e-6, nz - 1 - 1e-6]

    nlp = {'x': X, 'f': J, 'g': g}
    return nlp, lbx, ubx, lbg, ubg

# -----------------------------
# Main API: optimize_trajectory (callable)
# -----------------------------
def optimize_trajectory(env, start=(5.0,5.0,0.0), goal=(35.0,35.0,0.0), N=30,
                        w_acc=1.0, w_len=0.05, w_risk=0.5, margin=0.5,
                        ipopt_opts=None):
    """
    High-level function to run A*, resample, build CasADi NLP with SDF constraints, solve, and return results.
    Returns dict with keys:
      - X_opt: optimized nodes (N x 3)
      - X0: initial resampled nodes (N x 3)
      - coarse_path: list of voxel tuples from A*
      - occ, risk, sdf
      - solver_info: CasADi solver result
      - metrics: length/risk before/after
    """
    if ipopt_opts is None:
        ipopt_opts = {'ipopt.print_level': 3, 'print_time': False, 'ipopt.max_iter': 400, 'ipopt.tol': 1e-6}

    occ, risk, res = build_environment(env)
    nx, ny, nz = env["size"]
    start_vox = (int(start[0]), int(start[1]), int(start[2]))
    goal_vox = (int(goal[0]), int(goal[1]), int(goal[2]))

    coarse_path = astar_3d(start_vox, goal_vox, occ, risk, w_risk=8.0, base_cost=1.0)
    if coarse_path is None:
        raise RuntimeError("No coarse path found by A*.")

    X0 = resample_path(coarse_path, N)
    x0_flat = X0[1:-1].ravel()

    # compute SDF
    sdf = compute_sdf(occ)

    # build NLP
    nlp, lbx, ubx, lbg, ubg = build_casadi_nlp_with_sdf(
        start=np.array(start, dtype=float),
        goal=np.array(goal, dtype=float),
        N=N,
        env=env,
        sdf_np=sdf,
        w_acc=w_acc, w_len=w_len, w_risk=w_risk,
        margin=margin
    )

    solver = ca.nlpsol('solver', 'ipopt', nlp, ipopt_opts)
    sol = solver(x0=x0_flat, lbx=lbx, ubx=ubx, lbg=lbg, ubg=ubg)
    x_opt = sol['x'].full().ravel()

    X_opt = np.zeros((N, 3), dtype=float)
    X_opt[0] = start
    X_opt[-1] = goal
    X_opt[1:-1] = x_opt.reshape((N - 2, 3))

    # metrics
    segs_before = np.linalg.norm(np.diff(X0, axis=0), axis=1)
    segs_after = np.linalg.norm(np.diff(X_opt, axis=0), axis=1)
    len_before = segs_before.sum(); len_after = segs_after.sum()
    risk_before = sum(trilinear_interpolate(risk, p) for p in X0)
    risk_after = sum(trilinear_interpolate(risk, p) for p in X_opt)

    result = {
        'X_opt': X_opt,
        'X0': X0,
        'coarse_path': coarse_path,
        'occ': occ,
        'risk': risk,
        'sdf': sdf,
        'solver_info': sol,
        'metrics': {
            'len_before': float(len_before),
            'len_after': float(len_after),
            'risk_before': float(risk_before),
            'risk_after': float(risk_after)
        }
    }
    return result

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

    # Risk volume (sparse sampling)
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
# If run as script, quick demo
# -----------------------------
if __name__ == "__main__":
    env = default_environment()
    res = env.get("resolution", 1.0)
    w_acc = 1.0; w_len = 0.05; w_risk = 0.5; margin = 0.5
    result = optimize_trajectory(env, start=(10.0,5.0,0.0), goal=(35.0,35.0,10.0), N=30, w_acc=w_acc, w_len=w_len, w_risk=w_risk, margin=margin)
    print("Metrics:", result['metrics'])
    visualize_3d(result['occ'], result['risk'], result['X_opt'], coarse_nodes=result['X0'], res=res)
