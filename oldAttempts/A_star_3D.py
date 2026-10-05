import numpy as np
import heapq
import plotly.graph_objects as go

# -----------------------------
# Environment and risk generation
# -----------------------------

def create_3d_environment(nx=40, ny=40, nz=20, res=1.0):
    """
    Create a simple 3D voxel environment:
    - nx, ny, nz: grid dimensions
    - res: meters per voxel (not heavily used yet, but good to keep)
    Returns:
        occ: boolean occupancy grid [nz, ny, nx]
        risk: float risk grid [nz, ny, nx]
    """
    occ = np.zeros((nz, ny, nx), dtype=bool)
    risk = np.zeros((nz, ny, nx), dtype=np.float32)

    # Coordinate grids (center of voxels)
    xs = np.arange(nx) * res
    ys = np.arange(ny) * res
    zs = np.arange(nz) * res
    X, Y = np.meshgrid(xs, ys, indexing='xy')

    # --- Base 2D risk: a Gaussian blob in the center at ground level ---
    cx, cy = xs[3*nx // 4], ys[ny // 2]
    sigma = nx * res / 6.0
    base_risk_2d = np.exp(-((X - cx) ** 2 + (Y - cy) ** 2) / (2 * sigma ** 2))
    cx, cy = xs[nx // 4], ys[3*ny // 4]
    base_risk_2d_2 = np.exp(-((X - cx) ** 2 + (Y - cy) ** 2) / (2 * sigma ** 2))
    # --- Altitude modifier: risk decreases with altitude ---
    # f_z(z) = exp(-beta * z)
    beta = 0.1
    fz = np.exp(-beta * zs)  # shape [nz]

    # Combine into 3D risk: risk[z,y,x] = base_2d[y,x] * (1 + alpha * fz[z])
    alpha = 2.0
    for k in range(nz):
        risk[k, :, :] = (base_risk_2d + base_risk_2d_2) * (1.0 + alpha * fz[k])

    # Normalize risk to [0,1]
    risk /= risk.max() + 1e-8

    # --- Obstacles: a vertical column in the center and a wall ---
    # Column
    col_radius = nx // 8
    for k in range(nz):
        for j in range(ny):
            for i in range(nx):
                dx = i - nx // 2
                dy = j - ny // 2
                if dx * dx + dy * dy <= col_radius * col_radius and k < nz // 2:
                    occ[k, j, i] = True

    # Wall at some x
    wall_x = nx // 3
    for k in range(nz // 2):
        for j in range(ny // 3, 2 * ny // 3):
            occ[k, j, wall_x] = True

    return occ, risk, res


# -----------------------------
# 3D A* with risk-aware cost
# -----------------------------

def neighbors_3d(nx, ny, nz):
    """
    Precompute neighbor offsets and move costs for 26-connectivity in 3D.
    Returns list of (dx, dy, dz, move_cost).
    """
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


def astar_3d(start, goal, occ, risk, w_risk=5.0, base_cost=1.0, track_history=False):
    """
    3D A* on voxel grid with risk-aware cost.
    occ: [nz, ny, nx] boolean occupancy
    risk: [nz, ny, nx] float risk in [0,1]
    start, goal: (x, y, z) voxel indices
    w_risk: weight for risk cost
    base_cost: per-unit move cost scaling
    track_history: if True, return (path, history) instead of just path
    Returns:
        path: list of (x,y,z) from start to goal, or None if no path.
        history: (only if track_history=True) list of (visited_nodes, frontier_nodes) at each step
    """
    nz, ny, nx = occ.shape
    nbrs = neighbors_3d(nx, ny, nz)

    start_idx = idx_from_xyz(*start, nx, ny, nz)
    goal_idx = idx_from_xyz(*goal, nx, ny, nz)

    # g-costs and parent pointers
    g = np.full(nx * ny * nz, np.inf, dtype=np.float32)
    parent = np.full(nx * ny * nz, -1, dtype=np.int32)

    # Heuristic: Euclidean distance * base_cost (risk ignored for admissibility)
    def heuristic(x, y, z):
        gx, gy, gz = goal
        return base_cost * np.sqrt((x - gx) ** 2 + (y - gy) ** 2 + (z - gz) ** 2)

    # Priority queue: (f, idx)
    open_heap = []
    g[start_idx] = 0.0
    f_start = heuristic(*start)
    heapq.heappush(open_heap, (f_start, start_idx))

    in_open = np.zeros(nx * ny * nz, dtype=bool)
    in_open[start_idx] = True

    visited = set()
    history = []

    while open_heap:
        f_curr, curr_idx = heapq.heappop(open_heap)
        in_open[curr_idx] = False

        if curr_idx in visited:
            continue
        visited.add(curr_idx)

        if track_history:
            frontier = [xyz_from_idx(idx, nx, ny, nz) for idx, _ in open_heap]
            visited_xyz = [xyz_from_idx(idx, nx, ny, nz) for idx in visited]
            history.append((visited_xyz.copy(), frontier.copy()))

        if curr_idx == goal_idx:
            # Reconstruct path
            path = []
            idx = curr_idx
            while idx != -1:
                x, y, z = xyz_from_idx(idx, nx, ny, nz)
                path.append((x, y, z))
                idx = parent[idx]
            path.reverse()
            if track_history:
                return path, history
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

            if n_idx in visited:
                continue

            # Cost to move: base move cost + risk cost at neighbor
            step_cost = base_cost * move_cost + w_risk * risk[nz_, ny_, nx_]
            tentative_g = g[curr_idx] + step_cost

            if tentative_g < g[n_idx]:
                g[n_idx] = tentative_g
                parent[n_idx] = curr_idx
                f_n = tentative_g + heuristic(nx_, ny_, nz_)
                if not in_open[n_idx]:
                    heapq.heappush(open_heap, (f_n, n_idx))
                    in_open[n_idx] = True

    if track_history:
        return None, history
    return None  # no path


# -----------------------------
# Visualization with Plotly
# -----------------------------

def visualize_3d(occ, risk, path, res=1.0, output_file=None):
    """
    Visualize:
    - Obstacles as scatter points
    - Risk as semi-transparent volume
    - Path as 3D line
    
    Args:
        occ: occupancy grid
        risk: risk grid
        path: path to visualize
        res: resolution
        output_file: if provided, save as HTML file (e.g., 'visualization.html')
    """
    nz, ny, nx = occ.shape
    xs = np.arange(nx) * res
    ys = np.arange(ny) * res
    zs = np.arange(nz) * res

    # Obstacles
    occ_indices = np.argwhere(occ)
    if occ_indices.size > 0:
        ox = occ_indices[:, 2] * res
        oy = occ_indices[:, 1] * res
        oz = occ_indices[:, 0] * res
        obstacle_scatter = go.Scatter3d(
            x=ox, y=oy, z=oz,
            mode='markers',
            marker=dict(size=3, color='black'),
            name='Obstacles'
        )
    else:
        obstacle_scatter = None

    # Risk volume (downsample for speed if needed)
    # Plotly volume expects 3D array with x,y,z coordinates
    # We'll use full resolution here; for larger grids, consider downsampling.
    risk_volume = go.Volume(
        x=np.repeat(xs[np.newaxis, np.newaxis, :], ny, axis=1).repeat(nz, axis=0).flatten(),
        y=np.repeat(ys[np.newaxis, :, np.newaxis], nx, axis=2).repeat(nz, axis=0).flatten(),
        z=np.repeat(zs[:, np.newaxis, np.newaxis], ny, axis=1).repeat(nx, axis=2).flatten(),
        value=risk.flatten(),
        isomin=0.1,
        isomax=1.0,
        opacity=0.1,
        surface_count=15,
        colorscale='YlOrRd',
        name='Risk'
    )

    # Path line
    if path is not None and len(path) > 0:
        px = np.array([p[0] for p in path]) * res
        py = np.array([p[1] for p in path]) * res
        pz = np.array([p[2] for p in path]) * res
        path_line = go.Scatter3d(
            x=px, y=py, z=pz,
            mode='lines+markers',
            line=dict(color='blue', width=5),
            marker=dict(size=4, color='blue'),
            name='Path'
        )
    else:
        path_line = None

    data = [risk_volume]
    if obstacle_scatter is not None:
        data.append(obstacle_scatter)
    if path_line is not None:
        data.append(path_line)

    fig = go.Figure(data=data)
    fig.update_layout(
        scene=dict(
            xaxis_title='X',
            yaxis_title='Y',
            zaxis_title='Z',
            aspectmode='data'
        ),
        title='3D Risk-aware Path Planning'
    )
    
    # Save as HTML file if requested
    if output_file:
        fig.write_html(output_file)
        print(f"Visualization saved to: {output_file}")
    
    fig.show()


def animate_astar_3d_topdown(occ, risk, start, goal, res=1.0, w_risk=5.0, base_cost=1.0, 
                             frame_interval=50, downsample_history=1, output_file=None):
    """
    Animate the A* pathfinding process as a 2D top-down view (XY plane).
    Much more resource-efficient than 3D animation.
    
    Args:
        occ: [nz, ny, nx] boolean occupancy grid
        risk: [nz, ny, nx] float risk grid
        start: (x, y, z) start position
        goal: (x, y, z) goal position
        res: resolution (meters per voxel)
        w_risk: weight for risk cost
        base_cost: per-unit move cost scaling
        frame_interval: duration of each frame in milliseconds (lower = faster)
        downsample_history: only show every Nth history step (to reduce number of frames)
        output_file: if provided, save animation as HTML file (e.g., 'animation.html')
    
    Returns:
        fig: plotly figure with animation
    """
    # Run A* with history tracking
    path, history = astar_3d(start, goal, occ, risk, w_risk=w_risk, 
                             base_cost=base_cost, track_history=True)
    
    if path is None:
        print("No path found!")
        return None
    
    nz, ny, nx = occ.shape
    xs = np.arange(nx) * res
    ys = np.arange(ny) * res
    
    # Get obstacles at all Z levels (project onto XY plane)
    occ_indices = np.argwhere(occ)
    if occ_indices.size > 0:
        ox = occ_indices[:, 2] * res
        oy = occ_indices[:, 1] * res
    else:
        ox, oy = np.array([]), np.array([])
    
    # Get risk at ground level (z=0) or average across Z
    risk_xy = np.max(risk, axis=0)  # Take max risk across Z levels
    
    # Create frames for animation
    frames = []
    
    # Downsample history for performance
    history_downsampled = history[::downsample_history]
    
    for step_idx, (visited_nodes, frontier_nodes) in enumerate(history_downsampled):
        # Visited nodes (explored)
        if visited_nodes:
            vx = np.array([v[0] for v in visited_nodes]) * res
            vy = np.array([v[1] for v in visited_nodes]) * res
            visited_scatter = go.Scatter(
                x=vx, y=vy,
                mode='markers',
                marker=dict(size=4, color='lightblue', opacity=0.6),
                name='Visited'
            )
        else:
            visited_scatter = go.Scatter(
                x=[], y=[],
                mode='markers',
                marker=dict(size=4, color='lightblue'),
                name='Visited'
            )
        
        # Frontier nodes (open set)
        if frontier_nodes:
            fx = np.array([f[0] for f in frontier_nodes]) * res
            fy = np.array([f[1] for f in frontier_nodes]) * res
            frontier_scatter = go.Scatter(
                x=fx, y=fy,
                mode='markers',
                marker=dict(size=5, color='yellow', opacity=0.8),
                name='Frontier'
            )
        else:
            frontier_scatter = go.Scatter(
                x=[], y=[],
                mode='markers',
                marker=dict(size=5, color='yellow'),
                name='Frontier'
            )
        
        # Obstacles
        obstacle_scatter = go.Scatter(
            x=ox, y=oy,
            mode='markers',
            marker=dict(size=3, color='black', opacity=0.7),
            name='Obstacles'
        )
        
        # Start and goal
        start_scatter = go.Scatter(
            x=[start[0] * res], y=[start[1] * res],
            mode='markers',
            marker=dict(size=10, color='green', symbol='star'),
            name='Start'
        )
        
        goal_scatter = go.Scatter(
            x=[goal[0] * res], y=[goal[1] * res],
            mode='markers',
            marker=dict(size=10, color='red', symbol='star'),
            name='Goal'
        )
        
        frame = go.Frame(
            data=[visited_scatter, frontier_scatter, obstacle_scatter, start_scatter, goal_scatter],
            name=str(step_idx)
        )
        frames.append(frame)
    
    # Final frame with complete path
    px = np.array([p[0] for p in path]) * res
    py = np.array([p[1] for p in path]) * res
    path_line = go.Scatter(
        x=px, y=py,
        mode='lines+markers',
        line=dict(color='blue', width=3),
        marker=dict(size=4, color='blue'),
        name='Final Path'
    )
    
    # Obstacles
    obstacle_scatter_final = go.Scatter(
        x=ox, y=oy,
        mode='markers',
        marker=dict(size=3, color='black', opacity=0.7),
        name='Obstacles'
    )
    
    start_scatter_final = go.Scatter(
        x=[start[0] * res], y=[start[1] * res],
        mode='markers',
        marker=dict(size=10, color='green', symbol='star'),
        name='Start'
    )
    
    goal_scatter_final = go.Scatter(
        x=[goal[0] * res], y=[goal[1] * res],
        mode='markers',
        marker=dict(size=10, color='red', symbol='star'),
        name='Goal'
    )
    
    final_frame = go.Frame(
        data=[obstacle_scatter_final, path_line, start_scatter_final, goal_scatter_final],
        name='Final Path'
    )
    frames.append(final_frame)
    
    # Create initial data
    initial_visited = go.Scatter(
        x=[], y=[],
        mode='markers',
        marker=dict(size=4, color='lightblue'),
        name='Visited'
    )
    initial_frontier = go.Scatter(
        x=[], y=[],
        mode='markers',
        marker=dict(size=5, color='yellow'),
        name='Frontier'
    )
    
    # Create figure with initial frame
    fig = go.Figure(
        data=[initial_visited, initial_frontier, obstacle_scatter_final, start_scatter_final, goal_scatter_final],
        frames=frames
    )
    
    fig.update_layout(
        xaxis_title='X (m)',
        yaxis_title='Y (m)',
        title=f'2D Top-Down A* Pathfinding Animation (Path length: {len(path)} voxels, Explored: {len(history_downsampled[-1][0])})',
        hovermode='closest',
        updatemenus=[
            dict(
                type='buttons',
                showactive=False,
                buttons=[
                    dict(
                        label='▶ Play',
                        method='animate',
                        args=[None, {
                            'frame': {'duration': frame_interval, 'redraw': True},
                            'fromcurrent': True,
                            'transition': {'duration': 0}
                        }]
                    ),
                    dict(
                        label='⏸ Pause',
                        method='animate',
                        args=[[None], {
                            'frame': {'duration': 0, 'redraw': True},
                            'mode': 'immediate',
                            'transition': {'duration': 0}
                        }]
                    )
                ]
            )
        ],
        xaxis=dict(scaleanchor="y", scaleratio=1),
        yaxis=dict(scaleanchor="x", scaleratio=1),
    )
    
    # Save as HTML file if requested
    if output_file:
        fig.write_html(output_file)
        print(f"Animation saved to: {output_file}")
    
    return fig


# -----------------------------
# Demo scenario
# -----------------------------

def demo():
    occ, risk, res = create_3d_environment(nx=40, ny=40, nz=20, res=1.0)

    start = (5, 5, 5)          # near ground, corner
    goal = (35, 35, 0)         # opposite corner, ground

    path = astar_3d(start, goal, occ, risk, w_risk=50.0, base_cost=1.0)
    print("Path length (voxels):", len(path) if path is not None else "No path")

    visualize_3d(occ, risk, path, res=res, output_file='astar_visualization.html')


def demo_animation():
    """Demo showing the animated A* search process as a 2D top-down view."""
    occ, risk, res = create_3d_environment(nx=40, ny=40, nz=20, res=1.0)

    start = (5, 5, 5)          # near ground, corner
    goal = (35, 35, 0)         # opposite corner, ground

    # animate_astar_3d_topdown creates a lightweight 2D top-down animation
    fig = animate_astar_3d_topdown(
        occ, risk, start, goal, 
        res=res, 
        w_risk=10.0, 
        base_cost=1.0,
        frame_interval=50,         # 50ms per frame
        downsample_history=50,      # Show every 50th step
        output_file='astar_animation.html'  # Save as HTML file
    )
    
    if fig:
        fig.show()


if __name__ == "__main__":
    # Uncomment one of these to run:
    demo()           # Static visualization of final path
    # demo_animation()   # Animated search process
