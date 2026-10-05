import numpy as np
import heapq
from scipy.optimize import minimize
from scipy.optimize import NonlinearConstraint
import plotly.graph_objects as go

# -----------------------------
# Environment and risk generation (same as before)
# -----------------------------
def create_3d_environment(nx=40, ny=40, nz=20, res=1.0):
    occ = np.zeros((nz, ny, nx), dtype=bool)
    risk = np.zeros((nz, ny, nx), dtype=np.float32)

    xs = np.arange(nx) * res
    ys = np.arange(ny) * res
    zs = np.arange(nz) * res
    X, Y = np.meshgrid(xs, ys, indexing='xy')

    # Base 2D Gaussian risk 
    cx, cy = xs[3*nx // 4], ys[ny // 4]
    sigma = nx * res / 6.0
    base_risk_2d = np.exp(-((X - cx) ** 2 + (Y - cy) ** 2) / (2 * sigma ** 2))
    cx, cy = xs[nx // 4], ys[3*ny // 4]
    base_risk_2d_2 = np.exp(-((X - cx) ** 2 + (Y - cy) ** 2) / (2 * sigma ** 2))

    beta = 0.12
    fz = np.exp(-beta * zs)
    alpha = 2.0
    for k in range(nz):
        risk[k, :, :] = (base_risk_2d + base_risk_2d_2) * (1.0 + alpha * fz[k])
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
# A* (coarse planner) reused
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
                f_n = tentative_g + heuristic(nx_, ny, nz_)
                if not in_open[n_idx]:
                    heapq.heappush(open_heap, (f_n, n_idx))
                    in_open[n_idx] = True

    return None

# -----------------------------
# Utilities: resample and trilinear interpolation
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
# Objective and constraints for direct transcription
# -----------------------------
def build_trajectory_problem(risk_field, dt=1.0, w_acc=1.0, w_len=0.05, w_risk=6.0, w_climb=5.0, a_max=1.0, v_max=5.0):
    """
    Build objective and constraint functions for optimization.
    Variables: interior nodes positions and velocities for nodes 1..N-2.
    We'll pass start and goal (positions) and fix endpoint velocities to zero.
    """

    nz, ny, nx = risk_field.shape

    def unpack_vars(x_flat, N):
        """
        x_flat length = (N-2)*6 arranged as [p1x,p1y,p1z,v1x,v1y,v1z, p2x,...]
        returns X (N,3) and V (N,3) with endpoints filled externally by caller.
        """
        interior = x_flat.reshape((-1, 6))
        return interior[:, :3], interior[:, 3:6]

    def objective(x_flat, start, goal, N):
        # reconstruct full X and V
        interior_p, interior_v = unpack_vars(x_flat, N)
        X = np.zeros((N, 3), dtype=float)
        V = np.zeros((N, 3), dtype=float)
        X[0] = start['pos']; X[-1] = goal['pos']
        V[0] = start['vel']; V[-1] = goal['vel']
        X[1:-1] = interior_p
        V[1:-1] = interior_v

        # accelerations (finite difference)
        A = (V[1:] - V[:-1]) / dt  # shape (N-1,3)

        # acceleration cost (control effort)
        acc_cost = np.sum(np.sum(A ** 2, axis=1))

        # length cost
        segs = np.linalg.norm(np.diff(X, axis=0), axis=1)
        len_cost = np.sum(segs)

        # risk cost (sample risk at each node)
        risk_cost = 0.0
        for k in range(N):
            p = X[k]
            p_clamped = np.clip(p, [0, 0, 0], [nx - 1 - 1e-6, ny - 1 - 1e-6, nz - 1 - 1e-6])
            risk_cost += trilinear_interpolate(risk_field, p_clamped)

        # altitude energy cost: penalize upward motion (positive vertical velocity)
        # smooth positive part: pos(vz) = (vz + |vz|)/2
        vz = V[:, 2]
        pos_vz = 0.5 * (vz + np.abs(vz))
        climb_cost = np.sum(pos_vz ** 2) * dt

        J = w_acc * acc_cost + w_len * len_cost + w_risk * risk_cost + w_climb * climb_cost
        return J

    def dynamics_eq_constraints(x_flat, start, goal, N):
        """
        Enforce trapezoidal integration:
        X_{k+1} - X_k - 0.5*(V_k + V_{k+1})*dt = 0 for k=0..N-2
        We return a flat array of length 3*(N-1) (vectorized residuals).
        """
        interior_p, interior_v = unpack_vars(x_flat, N)
        X = np.zeros((N, 3), dtype=float)
        V = np.zeros((N, 3), dtype=float)
        X[0] = start['pos']; X[-1] = goal['pos']
        V[0] = start['vel']; V[-1] = goal['vel']
        X[1:-1] = interior_p
        V[1:-1] = interior_v

        res = []
        for k in range(N - 1):
            lhs = X[k + 1] - X[k] - 0.5 * (V[k] + V[k + 1]) * dt
            res.append(lhs)
        res = np.concatenate(res)  # length 3*(N-1)
        return res

    def accel_ineq_constraints(x_flat, start, goal, N):
        """
        Enforce acceleration magnitude <= a_max for each interval:
        || (V_{k+1} - V_k)/dt || - a_max <= 0  for k=0..N-2
        Return array of length (N-1) with values <= 0 when satisfied.
        """
        interior_p, interior_v = unpack_vars(x_flat, N)
        V = np.zeros((N, 3), dtype=float)
        V[0] = start['vel']; V[-1] = goal['vel']
        V[1:-1] = interior_v

        vals = []
        for k in range(N - 1):
            a_k = (V[k + 1] - V[k]) / dt
            vals.append(np.linalg.norm(a_k) - a_max)
        return np.array(vals)

    def velocity_bounds(N):
        """
        Return arrays lower and upper for velocities and positions for interior nodes.
        We'll use these to build box bounds for SLSQP.
        """
        n_interior = N - 2
        lower = []
        upper = []
        for _ in range(n_interior):
            # position bounds (x,y,z)
            lower.extend([0.0, 0.0, 0.0])
            upper.extend([nx - 1 - 1e-6, ny - 1 - 1e-6, nz - 1 - 1e-6])
            # velocity bounds (vx,vy,vz)
            lower.extend([-v_max, -v_max, -v_max])
            upper.extend([v_max, v_max, v_max])
        return np.array(lower), np.array(upper)

    return objective, dynamics_eq_constraints, accel_ineq_constraints, velocity_bounds

# -----------------------------
# Plotting with Plotly (same as before)
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
        data.append(go.Scatter3d(x=ox, y=oy, z=oz, mode='markers',
                                 marker=dict(size=3, color='black'), name='Obstacles'))

    Xg = np.repeat(xs[np.newaxis, np.newaxis, :], ny, axis=1).repeat(nz, axis=0).flatten()
    Yg = np.repeat(ys[np.newaxis, :, np.newaxis], nx, axis=2).repeat(nz, axis=0).flatten()
    Zg = np.repeat(zs[:, np.newaxis, np.newaxis], ny, axis=1).repeat(nx, axis=2).flatten()
    data.append(go.Volume(x=Xg, y=Yg, z=Zg, value=risk.flatten(),
                          isomin=0.05, isomax=1.0, opacity=0.12, surface_count=12,
                          colorscale='YlOrRd', name='Risk'))

    if path_nodes is not None:
        px = path_nodes[:, 0] * res
        py = path_nodes[:, 1] * res
        pz = path_nodes[:, 2] * res
        data.append(go.Scatter3d(x=px, y=py, z=pz, mode='lines+markers',
                                 line=dict(color='blue', width=5), marker=dict(size=4, color='blue'),
                                 name='Optimized Path'))

    fig = go.Figure(data=data)
    fig.update_layout(scene=dict(xaxis_title='X (m)', yaxis_title='Y (m)', zaxis_title='Z (m)', aspectmode='data'),
                      title=title)
    fig.show()

# -----------------------------
# Demo: run A*, resample, direct transcription with constraints
# -----------------------------
def demo_with_acc_limits():
    occ, risk, res = create_3d_environment(nx=40, ny=40, nz=20, res=1.0)

    start_pos = np.array([5.0, 5.0, 0.0])
    goal_pos = np.array([35.0, 35.0, 5.0])

    start_vox = (int(start_pos[0]), int(start_pos[1]), int(start_pos[2]))
    goal_vox = (int(goal_pos[0]), int(goal_pos[1]), int(goal_pos[2]))
    coarse_path = astar_3d(start_vox, goal_vox, occ, risk, w_risk=8.0, base_cost=1.0)
    if coarse_path is None:
        print("No coarse path found.")
        return
    print("Coarse path length (voxels):", len(coarse_path))

    N = 30
    X0 = resample_path(coarse_path, N)  # initial positions
    V0 = np.zeros_like(X0)  # initial velocities zero

    # Build problem
    dt = 1.0
    w_acc = 1.0
    w_len = 0.05
    w_risk = 6.0
    w_climb = 0.0
    a_max = 3  # max acceleration (voxels / s^2)
    v_max = 4.0  # max speed (voxels / s)

    objective, dyn_eq, accel_ineq, vel_bounds_fn = build_trajectory_problem(
        risk, dt=dt, w_acc=w_acc, w_len=w_len, w_risk=w_risk, w_climb=w_climb, a_max=a_max, v_max=v_max
    )

    # Variables: interior nodes only (1..N-2) each with p(3) and v(3)
    n_interior = N - 2
    x0_flat = np.hstack([X0[1:-1].ravel(), V0[1:-1].ravel()])  # length (N-2)*6

    # start and goal dicts with fixed endpoint velocities (zero)
    start = {'pos': start_pos, 'vel': np.array([0.0, 0.0, 0.0])}
    goal = {'pos': goal_pos, 'vel': np.array([0.0, 0.0, 0.0])}

    # Box bounds for positions and velocities
    lower, upper = vel_bounds_fn(N)
    bounds = [(float(lower[i]), float(upper[i])) for i in range(len(lower))]

    # Equality constraints for dynamics
    dyn_cons = {'type': 'eq',
                'fun': lambda x: dyn_eq(x, start, goal, N)}

    # Inequality constraints for acceleration limits (<=0 satisfied)
    accel_cons = {'type': 'ineq',
                  'fun': lambda x: -accel_ineq(x, start, goal, N)}  # SciPy expects fun(x) >= 0, so negate

    # Run SLSQP
    print("Starting constrained optimization (SLSQP)...")
    res_opt = minimize(lambda x: objective(x, start, goal, N),
                       x0_flat,
                       method='SLSQP',
                       bounds=bounds,
                       constraints=[dyn_cons, accel_cons],
                       options={'maxiter': 500, 'ftol': 1e-6, 'disp': True})

    if not res_opt.success:
        print("Optimization warning:", res_opt.message)

    # Reconstruct full trajectory
    interior = res_opt.x.reshape((-1, 6))
    X_opt = np.zeros((N, 3), dtype=float)
    V_opt = np.zeros((N, 3), dtype=float)
    X_opt[0] = start['pos']; X_opt[-1] = goal['pos']
    V_opt[0] = start['vel']; V_opt[-1] = goal['vel']
    X_opt[1:-1] = interior[:, :3]
    V_opt[1:-1] = interior[:, 3:6]

    # Metrics
    segs_before = np.linalg.norm(np.diff(X0, axis=0), axis=1)
    segs_after = np.linalg.norm(np.diff(X_opt, axis=0), axis=1)
    len_before = segs_before.sum()
    len_after = segs_after.sum()
    risk_before = sum(trilinear_interpolate(risk, p) for p in X0)
    risk_after = sum(trilinear_interpolate(risk, p) for p in X_opt)
    accs = np.linalg.norm((V_opt[1:] - V_opt[:-1]) / dt, axis=1)
    max_acc = accs.max()
    print(f"Length before: {len_before:.3f}, after: {len_after:.3f}")
    print(f"Risk before: {risk_before:.3f}, after: {risk_after:.3f}")
    print(f"Max acceleration after optimization: {max_acc:.3f} (limit {a_max})")

    visualize_3d(occ, risk, X_opt, res=res, title="Optimized 3D Trajectory with Acc Limits and Climb Cost")

if __name__ == "__main__":
    demo_with_acc_limits()
