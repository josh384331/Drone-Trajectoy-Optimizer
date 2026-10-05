"""
casadi_traj_opt_node_sdf_patched.py

CasADi + IPOPT trajectory optimizer (node-only SDF, robust mode).
Patched for older CasADi interpolant API (grid/data as Python lists).
Requires: casadi, numpy, scipy, plotly
Install with: conda install -c conda-forge casadi plotly scipy numpy
"""

import numpy as np
import heapq
from scipy.ndimage import distance_transform_edt
import casadi as ca
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# -----------------------------
# Environment and risk
# -----------------------------
def create_3d_environment(nx=40, ny=40, nz=20, res=1.0):
    occ = np.zeros((nz, ny, nx), dtype=bool)
    risk = np.zeros((nz, ny, nx), dtype=np.float32)

    xs = np.arange(nx) * res
    ys = np.arange(ny) * res
    zs = np.arange(nz) * res
    X, Y = np.meshgrid(xs, ys, indexing='xy')

    cx, cy = xs[nx // 2], ys[ny // 2]
    sigma = max(nx, ny) * res / 6.0
    base_risk_2d = np.exp(-((X - cx) ** 2 + (Y - cy) ** 2) / (2 * sigma ** 2))

    beta = 0.12
    fz = np.exp(-beta * zs)
    alpha = 2.0
    for k in range(nz):
        risk[k, :, :] = base_risk_2d * (1.0 + alpha * fz[k])

    risk /= (risk.max() + 1e-12)

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
# SDF
# -----------------------------
def compute_sdf(occ, res=1.0):
    occ_bool = occ.astype(bool)
    dist_out = distance_transform_edt(~occ_bool) * res
    dist_in = distance_transform_edt(occ_bool) * res
    sdf = dist_out - dist_in
    return sdf

# -----------------------------
# A* 3D
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
# Utilities
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

def trilinear_interpolate_numpy(field, p):
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
# CasADi NLP builder (node-only SDF) - patched for older interpolant API
# -----------------------------
EPS_V = 1e-6

def build_casadi_nlp(risk_field, sdf_field, grid_x, grid_y, grid_z,
                     dt=1.0, w_radial=1.0, w_len=0.05, w_risk=6.0, w_ground=8.0,
                     a_max_radial=1.5, v_target=1.5, w_speed=20.0, d_min=0.6):
    """
    Returns: build_for_N factory that constructs an NLP for a given N and endpoints.
    Uses older CasADi interpolant API: grid as list-of-lists, data as flat list.
    """

    # risk_field and sdf_field are (nz, ny, nx)
    # We need data shaped (nx, ny, nz) and flattened in Fortran order
    sdf_data_arr = np.transpose(sdf_field, (2, 1, 0))  # (nx, ny, nz)
    risk_data_arr = np.transpose(risk_field, (2, 1, 0))  # (nx, ny, nz)

    sdf_data_list = sdf_data_arr.flatten(order='F').tolist()
    risk_data_list = risk_data_arr.flatten(order='F').tolist()

    grid_x_list = np.asarray(grid_x, dtype=float).tolist()
    grid_y_list = np.asarray(grid_y, dtype=float).tolist()
    grid_z_list = np.asarray(grid_z, dtype=float).tolist()

    sdf_interp_template = ca.interpolant(
        'sdf_interp', 'linear',
        [grid_x_list, grid_y_list, grid_z_list],
        sdf_data_list,
        {}
    )

    risk_interp_template = ca.interpolant(
        'risk_interp', 'linear',
        [grid_x_list, grid_y_list, grid_z_list],
        risk_data_list,
        {}
    )

    def build_for_N(N, start_pos, goal_pos, start_vel, goal_vel):
        n_interior = max(0, N - 2)
        X_var = ca.MX.sym('Xvar', n_interior * 6)

        if n_interior > 0:
            mat = ca.reshape(X_var, n_interior, 6)
            P_interior = [ca.reshape(mat[i, 0:3], 3, 1) for i in range(n_interior)]
            V_interior = [ca.reshape(mat[i, 3:6], 3, 1) for i in range(n_interior)]
        else:
            P_interior = []
            V_interior = []

        X = [ca.reshape(ca.DM(start_pos), 3, 1)]
        V = [ca.reshape(ca.DM(start_vel), 3, 1)]
        for k in range(n_interior):
            X.append(P_interior[k])
            V.append(V_interior[k])
        X.append(ca.reshape(ca.DM(goal_pos), 3, 1))
        V.append(ca.reshape(ca.DM(goal_vel), 3, 1))

        J = 0
        radial_cost = 0
        for k in range(N - 1):
            v_k = V[k]; v_k1 = V[k + 1]
            a_k = (v_k1 - v_k) / dt
            v_mid = 0.5 * (v_k + v_k1)
            vm = ca.norm_2(v_mid)
            t_hat = v_mid / ca.fmax(vm, EPS_V)
            a_tang = ca.dot(a_k, t_hat) * t_hat
            a_perp = a_k - a_tang
            radial_cost += ca.dot(a_perp, a_perp)
        J += w_radial * radial_cost

        len_cost = 0
        for k in range(N - 1):
            len_cost += ca.norm_2(X[k + 1] - X[k])
        J += w_len * len_cost

        risk_cost = 0
        for k in range(N):
            pt = ca.vertcat(X[k][0], X[k][1], X[k][2])
            risk_cost += risk_interp_template(pt)
        J += w_risk * risk_cost

        gamma = 0.6
        ground_pen = 0
        for k in range(N):
            ground_pen += ca.exp(-gamma * X[k][2])
        J += w_ground * ground_pen

        speed_cost = 0
        for k in range(N):
            speed_cost += (ca.norm_2(V[k]) - v_target) ** 2
        J += w_speed * speed_cost

        g_eq_list = []
        for k in range(N - 1):
            g_eq_list.append(X[k + 1] - X[k] - 0.5 * (V[k] + V[k + 1]) * dt)

        g_ineq_list = []
        for k in range(N - 1):
            v_k = V[k]; v_k1 = V[k + 1]
            a_k = (v_k1 - v_k) / dt
            v_mid = 0.5 * (v_k + v_k1)
            vm = ca.norm_2(v_mid)
            t_hat = v_mid / ca.fmax(vm, EPS_V)
            a_tang = ca.dot(a_k, t_hat) * t_hat
            a_perp = a_k - a_tang
            g_ineq_list.append(a_max_radial - ca.norm_2(a_perp))

        for k in range(N):
            pt = ca.vertcat(X[k][0], X[k][1], X[k][2])
            g_ineq_list.append(sdf_interp_template(pt) - d_min)

        if len(g_eq_list) > 0:
            g_eq = ca.vertcat(*[ca.reshape(gi, -1, 1) for gi in g_eq_list])
        else:
            g_eq = ca.DM.zeros(0)

        if len(g_ineq_list) > 0:
            g_ineq_vec = ca.vertcat(*[ca.reshape(gi, -1, 1) for gi in g_ineq_list])
        else:
            g_ineq_vec = ca.DM.zeros(0)

        g_all = ca.vertcat(g_eq, g_ineq_vec)
        nlp = {'x': X_var, 'f': J, 'g': g_all}

        def build_initial_guess(X0, V0):
            if n_interior == 0:
                return np.array([])
            arr = np.zeros(n_interior * 6)
            for i in range(n_interior):
                arr[6 * i:6 * i + 3] = X0[i + 1]
                arr[6 * i + 3:6 * i + 6] = V0[i + 1]
            return arr

        g_eq_len = int(g_eq.size1()) if hasattr(g_eq, 'size1') and g_eq.size1() > 0 else 0
        g_ineq_len = int(g_ineq_vec.size1()) if hasattr(g_ineq_vec, 'size1') and g_ineq_vec.size1() > 0 else 0

        gl = np.concatenate([np.zeros(g_eq_len), -1e19 * np.ones(g_ineq_len)])
        gu = np.concatenate([np.zeros(g_eq_len),  1e19 * np.ones(g_ineq_len)])

        return nlp, build_initial_guess, (X_var.size1(), gl, gu)

    return build_for_N

# -----------------------------
# Plotting
# -----------------------------
def plot_panel_with_diagnostics(occ, risk, coarse_path, X_opt, V_opt, res=1.0, title="CasADi Trajectory"):
    nz, ny, nx = occ.shape
    xs = np.arange(nx) * res
    ys = np.arange(ny) * res
    zs = np.arange(nz) * res

    occ_indices = np.argwhere(occ)
    ox = occ_indices[:,2]*res if occ_indices.size>0 else np.array([])
    oy = occ_indices[:,1]*res if occ_indices.size>0 else np.array([])
    oz = occ_indices[:,0]*res if occ_indices.size>0 else np.array([])

    Xg = np.repeat(xs[np.newaxis, np.newaxis, :], ny, axis=1).repeat(nz, axis=0).flatten()
    Yg = np.repeat(ys[np.newaxis, :, np.newaxis], nx, axis=2).repeat(nz, axis=0).flatten()
    Zg = np.repeat(zs[:, np.newaxis, np.newaxis], ny, axis=1).repeat(nx, axis=2).flatten()

    fig = make_subplots(rows=3, cols=2,
                        specs=[[{"type":"scene","rowspan":3}, {"type":"xy"}],
                               [None, {"type":"xy"}],
                               [None, {"type":"xy"}]],
                        subplot_titles=(title, "Speed profile", "Tangential dot", "Radial accel"))

    fig.add_trace(go.Volume(x=Xg, y=Yg, z=Zg, value=risk.flatten(),
                            isomin=0.05, isomax=1.0, opacity=0.12,
                            surface_count=12, colorscale='YlOrRd',
                            colorbar=dict(
                                title='Risk',
                                x=1.02,
                                y=0.58,
                                len=0.68,
                                thickness=16,
                            ),
                            name='Risk'), row=1, col=1)

    if occ_indices.size>0:
        fig.add_trace(go.Scatter3d(x=ox, y=oy, z=oz, mode='markers',
                                   marker=dict(size=3,color='black'),
                                   name='Obstacles'), row=1, col=1)

    if coarse_path is not None:
        cp = np.array(coarse_path)
        fig.add_trace(go.Scatter3d(x=cp[:,0]*res, y=cp[:,1]*res, z=cp[:,2]*res,
                                   mode='lines+markers',
                                   line=dict(color='red', width=3, dash='dash'),
                                   marker=dict(size=2,color='red'),
                                   name='A* coarse'), row=1, col=1)

    fig.add_trace(go.Scatter3d(x=X_opt[:,0]*res, y=X_opt[:,1]*res, z=X_opt[:,2]*res,
                               mode='lines+markers',
                               line=dict(color='blue', width=5),
                               marker=dict(size=4,color='blue'),
                               name='Optimized'), row=1, col=1)

    t = np.arange(X_opt.shape[0])
    speeds = np.linalg.norm(V_opt, axis=1)
    fig.add_trace(go.Scatter(x=t, y=speeds, mode='lines+markers', name='|v|'),
                  row=1, col=2)
    fig.update_xaxes(title_text="node index", row=1, col=2)
    fig.update_yaxes(title_text="speed", row=1, col=2)

    tang_dots = []
    radial_accs = []
    dt = 1.0
    for k in range(X_opt.shape[0]-1):
        a_k = (V_opt[k+1]-V_opt[k])/dt
        v_mid = 0.5*(V_opt[k]+V_opt[k+1])
        vm = np.linalg.norm(v_mid)
        if vm>1e-6:
            t_hat = v_mid/vm
        else:
            dp = X_opt[k+1]-X_opt[k]; nd=np.linalg.norm(dp)
            t_hat = dp/nd if nd>1e-9 else np.array([1.0,0.0,0.0])
        tang = float(np.dot(a_k, t_hat))
        tang_dots.append(tang)
        a_tang = tang*t_hat
        a_perp = a_k - a_tang
        radial_accs.append(np.linalg.norm(a_perp))

    fig.add_trace(go.Scatter(x=np.arange(len(tang_dots)), y=tang_dots,
                             mode='lines+markers', name='a·t_hat'),
                  row=2, col=2)
    fig.update_xaxes(title_text="interval index", row=2, col=2)
    fig.update_yaxes(title_text="tangential dot", row=2, col=2)

    fig.add_trace(go.Bar(x=np.arange(len(radial_accs)), y=radial_accs,
                         name='|a_perp|'),
                  row=3, col=2)
    fig.update_xaxes(title_text="interval index", row=3, col=2)
    fig.update_yaxes(title_text="radial accel", row=3, col=2)

    fig.update_layout(scene=dict(xaxis_title='X', yaxis_title='Y', zaxis_title='Z',
                                 aspectmode='data'),
                      height=1000, width=1300,
                      legend=dict(
                          orientation='h',
                          yanchor='bottom',
                          y=1.02,
                          xanchor='left',
                          x=0.0,
                          bgcolor='rgba(255,255,255,0.75)',
                          bordercolor='rgba(0,0,0,0.15)',
                          borderwidth=1,
                      ),
                      margin=dict(l=30, r=160, t=90, b=40))
    fig.show()

# -----------------------------
# Demo: build and solve with CasADi IPOPT (patched)
# -----------------------------
def demo_casadi_node_sdf():
    occ, risk, res = create_3d_environment(nx=40, ny=40, nz=20, res=1.0)
    sdf = compute_sdf(occ, res=res)

    nx_g = occ.shape[2]; ny_g = occ.shape[1]; nz_g = occ.shape[0]
    xs = np.arange(nx_g)
    ys = np.arange(ny_g)
    zs = np.arange(nz_g)

    start_pos = np.array([5.0, 5.0, 15.0])
    goal_pos  = np.array([35.0, 35.0, 0.0])
    start_vox = (int(start_pos[0]), int(start_pos[1]), int(start_pos[2]))
    goal_vox  = (int(goal_pos[0]),  int(goal_pos[1]),  int(goal_pos[2]))

    coarse_path = astar_3d(start_vox, goal_vox, occ, risk, w_risk=8.0, base_cost=1.0)
    if coarse_path is None:
        print("No coarse path found."); return
    print("Coarse path length:", len(coarse_path))

    N = 20
    X0 = resample_path(coarse_path, N)

    dt = 1.0
    V0 = np.zeros_like(X0)
    for k in range(X0.shape[0] - 1):
        dp = X0[k+1] - X0[k]
        V0[k]   = dp / dt
        V0[k+1] = dp / dt

    build_for_N = build_casadi_nlp(risk, sdf, xs, ys, zs,
                                   dt=dt, w_radial=1.0, w_len=0.05, w_risk=6.0, w_ground=1.0,
                                   a_max_radial=6, v_target=2, w_speed=0.1, d_min=0.6)

    nlp, build_initial_guess, (nx_var, gl, gu) = build_for_N(N, start_pos, goal_pos, V0[0], V0[-1])

    opts = {
        'ipopt.print_level': 5,
        'ipopt.max_iter': 1000,
        'ipopt.tol': 1e-6,
        'print_time': False
    }

    solver = ca.nlpsol('solver', 'ipopt', nlp, opts)

    x0 = build_initial_guess(X0, V0)

    v_max = 4.0
    lower = []
    upper = []
    n_interior = max(0, N - 2)
    for _ in range(n_interior):
        lower.extend([0.0, 0.0, 0.0]); upper.extend([nx_g - 1 - 1e-6, ny_g - 1 - 1e-6, nz_g - 1 - 1e-6])
        lower.extend([-v_max, -v_max, -v_max]); upper.extend([v_max, v_max, v_max])
    lbx = np.array(lower)
    ubx = np.array(upper)

    if x0.size > 0:
        sdf_vals_init = [trilinear_interpolate_numpy(sdf, X0[k]) for k in range(N)]
        print("Initial min SDF at nodes (x0):", min(sdf_vals_init))

    print("Starting IPOPT solve...")
    sol = solver(x0=x0, lbx=lbx, ubx=ubx, lbg=gl, ubg=gu)

    x_opt_ca = sol['x']
    x_opt = np.array(x_opt_ca.full()).flatten() if x_opt_ca.size1() > 0 else np.array([])
    if x_opt.size:
        interior = x_opt.reshape((n_interior, 6))
    else:
        interior = np.zeros((0, 6))

    X_opt = np.zeros((N, 3)); V_opt = np.zeros((N, 3))
    X_opt[0] = start_pos; X_opt[-1] = goal_pos
    V_opt[0] = V0[0]; V_opt[-1] = V0[-1]
    if N > 2:
        X_opt[1:-1] = interior[:, :3]
        V_opt[1:-1] = interior[:, 3:6]

    segs_before = np.linalg.norm(np.diff(X0, axis=0), axis=1)
    segs_after = np.linalg.norm(np.diff(X_opt, axis=0), axis=1)
    print(f"Length before: {segs_before.sum():.3f}, after: {segs_after.sum():.3f}")

    accs = np.linalg.norm((V_opt[1:] - V_opt[:-1]) / dt, axis=1)
    print(f"Max accel: {accs.max():.3f} (radial limit ~1.5)")

    min_sdf = min(trilinear_interpolate_numpy(sdf, p) for p in X_opt)
    print(f"Min SDF along nodes: {min_sdf:.3f}")

    print("Plotting results...")
    plot_panel_with_diagnostics(occ, risk, coarse_path, X_opt, V_opt, res=res, title="CasADi IPOPT Node SDF (patched)")

if __name__ == "__main__":
    demo_casadi_node_sdf()


# this was way faster that the original implimentation but we may have overconstrained the problem.  let's go back to the original A* and just make the path smooth with CASADI