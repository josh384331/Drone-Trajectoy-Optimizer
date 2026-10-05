import numpy as np
import heapq
from scipy.optimize import minimize
from scipy.ndimage import distance_transform_edt
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
# Robust problem builder
# -----------------------------
EPS_V = 1e-3

def build_problem_robust(risk_field, sdf_field, dt=1.0,
                         w_radial=1.0, w_len=0.05, w_risk=6.0, w_ground=8.0,
                         a_max_radial=1.5, v_target=1.5, w_speed=20.0):
    nz, ny, nx = risk_field.shape

    def unpack(x_flat):
        interior = x_flat.reshape((-1, 6)) if x_flat.size else np.zeros((0,6))
        return interior[:, :3], interior[:, 3:6]

    def objective(x_flat, start, goal, N):
        interior_p, interior_v = unpack(x_flat)
        X = np.zeros((N,3)); V = np.zeros((N,3))
        X[0]=start['pos']; X[-1]=goal['pos']
        V[0]=start['vel']; V[-1]=goal['vel']
        if N>2:
            X[1:-1]=interior_p; V[1:-1]=interior_v

        A = (V[1:] - V[:-1]) / dt
        radial_cost = 0.0
        for k in range(A.shape[0]):
            v_mid = 0.5*(V[k]+V[k+1])
            vm = np.linalg.norm(v_mid)
            if vm > EPS_V:
                t_hat = v_mid / vm
            else:
                dp = X[k+1]-X[k]; nd = np.linalg.norm(dp)
                t_hat = dp/nd if nd>1e-9 else np.array([1.0,0.0,0.0])
            a_k = A[k]
            a_tang = np.dot(a_k, t_hat)*t_hat
            a_perp = a_k - a_tang
            radial_cost += np.dot(a_perp, a_perp)

        len_cost = np.sum(np.linalg.norm(np.diff(X,axis=0),axis=1))

        risk_cost = sum(trilinear_interpolate(risk_field,
                                              np.clip(X[k],[0,0,0],[nx-1-1e-6,ny-1-1e-6,nz-1-1e-6]))
                        for k in range(N))

        gamma = 0.6
        ground_pen = np.sum(np.exp(-gamma * X[:,2]))

        speeds = np.linalg.norm(V, axis=1)
        speed_cost = np.sum((speeds - v_target)**2)

        J = (w_radial*radial_cost + w_len*len_cost +
             w_risk*risk_cost + w_ground*ground_pen +
             w_speed*speed_cost)
        return J

    def dynamics_eq(x_flat, start, goal, N):
        interior_p, interior_v = unpack(x_flat)
        X = np.zeros((N,3)); V = np.zeros((N,3))
        X[0]=start['pos']; X[-1]=goal['pos']
        V[0]=start['vel']; V[-1]=goal['vel']
        if N>2:
            X[1:-1]=interior_p; V[1:-1]=interior_v
        res=[]
        for k in range(N-1):
            res.append(X[k+1]-X[k]-0.5*(V[k]+V[k+1])*dt)
        return np.concatenate(res)

    def radial_mag_ineq(x_flat, start, goal, N):
        interior_p, interior_v = unpack(x_flat)
        V = np.zeros((N,3)); V[0]=start['vel']; V[-1]=goal['vel']
        if N>2:
            V[1:-1]=interior_v
        vals=[]
        for k in range(N-1):
            a_k = (V[k+1]-V[k])/dt
            v_mid = 0.5*(V[k]+V[k+1])
            vm = np.linalg.norm(v_mid)
            if vm > EPS_V:
                t_hat = v_mid/vm
            else:
                t_hat = np.array([1.0,0.0,0.0])
            a_tang = np.dot(a_k, t_hat)*t_hat
            a_perp = a_k - a_tang
            vals.append(a_max_radial - np.linalg.norm(a_perp))
        return np.array(vals)

    def bounds_for_interior(N, v_max_local):
        n_interior = N-2
        lower=[]; upper=[]
        for _ in range(n_interior):
            lower.extend([0.0,0.0,0.0]); upper.extend([nx-1-1e-6, ny-1-1e-6, nz-1-1e-6])
            lower.extend([-v_max_local, -v_max_local, -v_max_local])
            upper.extend([v_max_local, v_max_local, v_max_local])
        return np.array(lower), np.array(upper)

    return objective, dynamics_eq, radial_mag_ineq, bounds_for_interior

# -----------------------------
# Node SDF constraint
# -----------------------------
def sdf_node_ineq(x_flat, start, goal, N, sdf_field):
    interior = x_flat.reshape((-1,6)) if x_flat.size else np.zeros((0,6))
    interior_p = interior[:,:3] if interior.size else np.zeros((0,3))
    X = np.zeros((N,3)); X[0]=start['pos']; X[-1]=goal['pos']
    if N>2:
        X[1:-1]=interior_p
    nz, ny, nx = sdf_field.shape
    vals=[]
    for k in range(N):
        p_clamped = np.clip(X[k],[0,0,0],[nx-1-1e-6,ny-1-1e-6,nz-1-1e-6])
        vals.append(trilinear_interpolate(sdf_field, p_clamped) - 0.6)
    return np.array(vals)

# -----------------------------
# Plotting
# -----------------------------
def plot_panel_with_diagnostics(occ, risk, coarse_path, X_opt, V_opt, res=1.0, title="Robust Trajectory"):
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
        if vm>EPS_V:
            t_hat = v_mid/vm
        else:
            dp = X_opt[k+1]-X_opt[k]; nd=np.linalg.norm(dp)
            t_hat = dp/nd if nd>1e-9 else np.array([1.0,0.0,0.0])
        tang = np.dot(a_k, t_hat)
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
                      height=1000, width=1200)
    fig.show()

# -----------------------------
# Demo
# -----------------------------
def demo_robust():
    occ, risk, res = create_3d_environment(nx=40, ny=40, nz=20, res=1.0)
    sdf = compute_sdf(occ, res=res)

    start_pos = np.array([5.0, 5.0, 10.0])
    goal_pos  = np.array([35.0, 35.0, 10.0])
    start_vox = (int(start_pos[0]), int(start_pos[1]), int(start_pos[2]))
    goal_vox  = (int(goal_pos[0]),  int(goal_pos[1]),  int(goal_pos[2]))

    coarse_path = astar_3d(start_vox, goal_vox, occ, risk, w_risk=8.0, base_cost=1.0)
    if coarse_path is None:
        print("No coarse path found.")
        return

    print("Coarse path length:", len(coarse_path))

    N = 20
    X0 = resample_path(coarse_path, N)

    dt = 1.0
    V0 = np.zeros_like(X0)
    for k in range(X0.shape[0]-1):
        dp = X0[k+1] - X0[k]
        V0[k]   = dp / dt
        V0[k]*= 0.8
        V0[k+1] = dp / dt
        

    objective, dyn_eq, radial_mag_ineq, bounds_fn = build_problem_robust(
        risk, sdf, dt=dt,
        w_radial=1.0, w_len=0.05, w_risk=6.0, w_ground=1.0,
        a_max_radial=5, v_target=1.5, w_speed=1.0
    )

    x0_flat = np.hstack([X0[1:-1].ravel(), V0[1:-1].ravel()]) if N>2 else np.array([])
    start = {'pos': start_pos, 'vel': V0[0]}
    goal  = {'pos': goal_pos,  'vel': V0[-1]}

    lower, upper = bounds_fn(N, v_max_local=4.0)
    bounds = [(float(lower[i]), float(upper[i])) for i in range(len(lower))]

    dyn_cons   = {'type':'eq',   'fun': lambda x: dyn_eq(x, start, goal, N)}
    radial_cons= {'type':'ineq', 'fun': lambda x: radial_mag_ineq(x, start, goal, N)}
    sdf_cons   = {'type':'ineq', 'fun': lambda x: sdf_node_ineq(x, start, goal, N, sdf)}

    constraints = [dyn_cons, radial_cons, sdf_cons]

    print("Starting robust optimization (SLSQP)...")
    res_opt = minimize(lambda x: objective(x, start, goal, N),
                       x0_flat, method='SLSQP', bounds=bounds, constraints=constraints,
                       options={'maxiter':800, 'ftol':1e-6, 'disp':True})

    if not res_opt.success:
        print("Optimization warning:", res_opt.message)

    interior = res_opt.x.reshape((-1,6)) if res_opt.x.size else np.zeros((0,6))
    X_opt = np.zeros((N,3)); V_opt = np.zeros((N,3))
    X_opt[0]=start['pos']; X_opt[-1]=goal['pos']
    V_opt[0]=start['vel']; V_opt[-1]=goal['vel']
    if N>2:
        X_opt[1:-1]=interior[:,:3]
        V_opt[1:-1]=interior[:,3:6]

    segs_before = np.linalg.norm(np.diff(X0,axis=0),axis=1)
    segs_after  = np.linalg.norm(np.diff(X_opt,axis=0),axis=1)
    print(f"Length before: {segs_before.sum():.3f}, after: {segs_after.sum():.3f}")

    accs = np.linalg.norm((V_opt[1:]-V_opt[:-1])/dt,axis=1)
    print(f"Max accel: {accs.max():.3f} (limit ~{1.5})")

    min_sdf = min(trilinear_interpolate(sdf, np.clip(p,[0,0,0],[sdf.shape[2]-1-1e-6,
                                                                sdf.shape[1]-1-1e-6,
                                                                sdf.shape[0]-1-1e-6]))
                  for p in X_opt)
    print(f"Min SDF along nodes: {min_sdf:.3f}")

    plot_panel_with_diagnostics(occ, risk, coarse_path, X_opt, V_opt, res=res,
                                title="Robust Trajectory (node SDF, speed penalty)")

if __name__ == "__main__":
    demo_robust()
