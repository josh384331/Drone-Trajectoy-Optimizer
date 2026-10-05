import numpy as np
import heapq
import casadi as ca
import plotly.graph_objects as go

# -----------------------------
# Environment and risk generation
# -----------------------------
def create_3d_environment(nx=40, ny=40, nz=20, res=1.0):
    """
    Create occupancy grid and altitude-dependent risk field.
    occ: boolean array [nz, ny, nx]
    risk: float array [nz, ny, nx] normalized to [0,1]
    res: meters per voxel
    """
    occ = np.zeros((nz, ny, nx), dtype=bool)
    risk = np.zeros((nz, ny, nx), dtype=np.float32)

    xs = np.arange(nx) * res
    ys = np.arange(ny) * res
    zs = np.arange(nz) * res
    X, Y = np.meshgrid(xs, ys, indexing='xy')

    # Base 2D Gaussian risk centered in map
    cx, cy = xs[3 * nx // 4], ys[ny // 2]
    sigma = nx * res / 6.0
    base_risk_2d = np.exp(-((X - cx) ** 2 + (Y - cy) ** 2) / (2 * sigma ** 2))
    cx, cy = xs[nx // 4], ys[3 * ny // 4]
    base_risk_2d_2 = np.exp(-((X - cx) ** 2 + (Y - cy) ** 2) / (2 * sigma ** 2))

    # Altitude modifier: risk decreases with altitude
    beta = 0.12
    fz = np.exp(-beta * zs)  # shape [nz]

    alpha = 2.0  # altitude effect scale
    for k in range(nz):
        risk[k, :, :] = (base_risk_2d + base_risk_2d_2) * (1.0 + alpha * fz[k])

    # Normalize risk
    risk /= (risk.max() + 1e-12)

    # Obstacles: central vertical column and a low wall
    col_radius = nx // 8
    for k in range(nz):
        for j in range(ny):
            for i in range(nx):
                dx = i - nx // 2
                dy = j - ny // 2
                if dx * dx + dy * dy <= col_radius * col_radius and k < nz // 2:
                    occ[k, j, i] = True

    wall_x = nx // 3
    for k in range(nz // 2):
        for j in range(ny // 3, 2 * ny // 3):
            occ[k, j, wall_x] = True

    return occ, risk, res

# -----------------------------
# 3D A*
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
    return x + nx * (y + ny * z)

def xyz_from_idx(idx, nx, ny, nz):
    z = idx // (nx * ny)
    rem = idx % (nx * ny)
    y = rem // nx
    x = rem % nx
    return x, y, z

def astar_3d(start, goal, occ, risk, w_risk=5.0, base_cost=1.0):
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
            step_cost = base_cost * move_cost + w_risk * risk[nz_, ny_, nx_]
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
# Utilities: resample path and trilinear interpolation of risk
# -----------------------------
def resample_path(path_voxels, num_nodes):
    """
    Convert voxel indices to continuous coordinates and resample to num_nodes points
    using linear interpolation along cumulative distance.
    path_voxels: list of (x,y,z) voxel indices
    returns: np.array shape (num_nodes, 3) in voxel coordinates (not meters)
    """
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
    """
    Trilinear interpolation of 3D field at continuous voxel coordinates p = (x,y,z).
    field shape: [nz, ny, nx]
    returns scalar
    """
    nz, ny, nx = field.shape
    x, y, z = p
    x = np.clip(x, 0, nx - 1 - 1e-6)
    y = np.clip(y, 0, ny - 1 - 1e-6)
    z = np.clip(z, 0, nz - 1 - 1e-6)

    x0 = int(np.floor(x)); x1 = x0 + 1
    y0 = int(np.floor(y)); y1 = y0 + 1
    z0 = int(np.floor(z)); z1 = z0 + 1

    xd = x - x0; yd = y - y0; zd = z - z0

    c000 = field[z0, y0, x0]
    c100 = field[z0, y0, x1]
    c010 = field[z0, y1, x0]
    c110 = field[z0, y1, x1]
    c001 = field[z1, y0, x0]
    c101 = field[z1, y0, x1]
    c011 = field[z1, y1, x0]
    c111 = field[z1, y1, x1]

    c00 = c000 * (1 - xd) + c100 * xd
    c01 = c001 * (1 - xd) + c101 * xd
    c10 = c010 * (1 - xd) + c110 * xd
    c11 = c011 * (1 - xd) + c111 * xd

    c0 = c00 * (1 - yd) + c10 * yd
    c1 = c01 * (1 - yd) + c11 * yd

    c = c0 * (1 - zd) + c1 * zd
    return float(c)

# -----------------------------
# CasADi NLP objective builder
# -----------------------------
def build_casadi_nlp(start, goal, N,
                     nx, ny, nz, res,
                     w_acc=1.0, w_len=0.1, w_risk=5.0):
    """
    Build CasADi NLP for interior nodes (start and goal fixed).
    Risk is computed analytically (same as create_3d_environment),
    so we avoid DM/interpolant issues.
    """

    # decision variables: interior nodes flattened
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

    # ----------------------------------------
    # Analytic risk function (CasADi SX)
    # same structure as create_3d_environment
    # ----------------------------------------
    beta = 0.12
    alpha = 2.0
    sigma = nx * res / 6.0

    cx1 = 3 * nx * res / 4.0
    cy1 = ny * res / 2.0
    cx2 = nx * res / 4.0
    cy2 = 3 * ny * res / 4.0

    def risk_analytic(p):
        # p in voxel coords (x,y,z)
        x = p[0] * res
        y = p[1] * res
        z = p[2] * res

        dx1 = x - cx1
        dy1 = y - cy1
        dx2 = x - cx2
        dy2 = y - cy2

        g1 = ca.exp(-(dx1*dx1 + dy1*dy1) / (2 * sigma * sigma))
        g2 = ca.exp(-(dx2*dx2 + dy2*dy2) / (2 * sigma * sigma))

        fz = ca.exp(-beta * z)
        r = (g1 + g2) * (1.0 + alpha * fz)

        # optional normalization factor (approximate)
        # just to keep magnitudes similar to grid version
        return r

    # ----------------------------------------
    # Build objective
    # ----------------------------------------
    J = 0

    # acceleration cost
    for k in range(1, N - 1):
        x_km1 = node_sym(k - 1)
        x_k = node_sym(k)
        x_kp1 = node_sym(k + 1)
        acc = x_kp1 - 2 * x_k + x_km1
        J += w_acc * ca.sumsqr(acc)

    # length cost
    for k in range(N - 1):
        p0 = node_sym(k)
        p1 = node_sym(k + 1)
        seg = p1 - p0
        J += w_len * ca.norm_2(seg)

    # risk cost (analytic)
    for k in range(N):
        p = node_sym(k)
        # clamp voxel coords inside grid
        px = ca.fmax(0, ca.fmin(p[0], nx - 1 - 1e-6))
        py = ca.fmax(0, ca.fmin(p[1], ny - 1 - 1e-6))
        pz = ca.fmax(0, ca.fmin(p[2], nz - 1 - 1e-6))
        J += w_risk * risk_analytic(ca.vertcat(px, py, pz))

    # bounds for interior nodes
    lbx = []
    ubx = []
    for i in range(n_interior):
        lbx += [0.0, 0.0, 0.0]
        ubx += [nx - 1 - 1e-6, ny - 1 - 1e-6, nz - 1 - 1e-6]

    nlp = {'x': X, 'f': J}
    return nlp, lbx, ubx


# -----------------------------
# Plotting with Plotly
# -----------------------------
def visualize_3d(occ, risk, path_nodes, res=1.0, title="Trajectory Optimization"):
    nz, ny, nx = occ.shape
    xs = np.arange(nx) * res
    ys = np.arange(ny) * res
    zs = np.arange(nz) * res

    occ_indices = np.argwhere(occ)
    data = []
    if occ_indices.size > 0:
        ox = occ_indices[:, 2] * res
        oy = occ_indices[:, 1] * res
        oz = occ_indices[:, 0] * res
        data.append(go.Scatter3d(
            x=ox, y=oy, z=oz, mode='markers',
            marker=dict(size=3, color='black'), name='Obstacles'
        ))

    Xg = np.repeat(xs[np.newaxis, np.newaxis, :], ny, axis=1).repeat(nz, axis=0).flatten()
    Yg = np.repeat(ys[np.newaxis, :, np.newaxis], nx, axis=2).repeat(nz, axis=0).flatten()
    Zg = np.repeat(zs[:, np.newaxis, np.newaxis], ny, axis=1).repeat(nx, axis=2).flatten()
    data.append(go.Volume(
        x=Xg, y=Yg, z=Zg, value=risk.flatten(),
        isomin=0.05, isomax=1.0, opacity=0.12, surface_count=12,
        colorscale='YlOrRd', name='Risk'
    ))

    if path_nodes is not None:
        px = path_nodes[:, 0] * res
        py = path_nodes[:, 1] * res
        pz = path_nodes[:, 2] * res
        data.append(go.Scatter3d(
            x=px, y=py, z=pz, mode='lines+markers',
            line=dict(color='blue', width=5),
            marker=dict(size=4, color='blue'),
            name='Optimized Path'
        ))

    fig = go.Figure(data=data)
    fig.update_layout(
        scene=dict(
            xaxis_title='X (m)',
            yaxis_title='Y (m)',
            zaxis_title='Z (m)',
            aspectmode='data'
        ),
        title=title
    )
    fig.show()
    uniqueDateTime = np.datetime64('now').astype(str).replace(':', '-').replace(' ', '_')
    fig.write_html(f"optimized_trajectory_{uniqueDateTime}.html")

# -----------------------------
# Demo: run A*, resample, optimize (CasADi), visualize
# -----------------------------
def demo_optimize():
    occ, risk, res = create_3d_environment(nx=40, ny=40, nz=20, res=1.0)

    start = (5.0, 5.0, 0.0)
    goal = (35.0, 35.0, 0.0)

    start_vox = (int(start[0]), int(start[1]), int(start[2]))
    goal_vox = (int(goal[0]), int(goal[1]), int(goal[2]))
    coarse_path = astar_3d(start_vox, goal_vox, occ, risk, w_risk=8.0, base_cost=1.0)
    if coarse_path is None:
        print("No coarse path found.")
        return
    print("Coarse path length (voxels):", len(coarse_path))

    N = 60
    X0 = resample_path(coarse_path, N)

    x0_flat = X0[1:-1].ravel()

    w_acc = 0.5
    w_len = 0.05
    w_risk = 0.2

    nz, ny, nx = occ.shape  # note: occ.shape = (nz, ny, nx)

    nlp, lbx, ubx = build_casadi_nlp(
        start=np.array(start, dtype=float),
        goal=np.array(goal, dtype=float),
        N=N,
        nx=nx, ny=ny, nz=nz, res=res,
        w_acc=w_acc,
        w_len=w_len,
        w_risk=w_risk
    )


    opts = {
        'ipopt.print_level': 3,
        'print_time': False,
        'ipopt.max_iter': 400,
        'ipopt.tol': 1e-6
    }
    solver = ca.nlpsol('solver', 'ipopt', nlp, opts)

    print("Starting CasADi/IPOPT optimization...")
    sol = solver(x0=x0_flat, lbx=lbx, ubx=ubx)
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

    visualize_3d(occ, risk, X_opt, res=res,
                 title="Optimized 3D Trajectory (CasADi double-integrator cost)")

if __name__ == "__main__":
    demo_optimize()
