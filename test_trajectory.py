# tests/test_trajectory.py
import numpy as np
import pytest
from trajectory_module import default_environment, build_environment, compute_sdf, optimize_trajectory

@pytest.fixture(scope="module")
def env_and_basic():
    env = default_environment()
    occ, risk, res = build_environment(env)
    sdf = compute_sdf(occ)
    return {"env": env, "occ": occ, "risk": risk, "res": res, "sdf": sdf}

def test_optimize_runs(env_and_basic):
    env = env_and_basic["env"]
    # Use a small node count to keep test fast
    result = optimize_trajectory(env, start=(5.0,5.0,0.0), goal=(35.0,35.0,0.0), N=20,
                                w_acc=1.0, w_len=0.05, w_risk=1.0, margin=0.5,
                                ipopt_opts={'ipopt.print_level': 0, 'print_time': False, 'ipopt.max_iter': 200})
    # Basic structure checks
    assert "X_opt" in result
    assert "X0" in result
    assert "occ" in result
    assert "risk" in result
    assert result["X_opt"].shape[0] == 20
    assert result["X_opt"].shape[1] == 3

def test_no_collision_with_margin(env_and_basic):
    env = env_and_basic["env"]
    occ = env_and_basic["occ"]
    sdf = env_and_basic["sdf"]
    margin = 0.5
    # Run optimizer
    result = optimize_trajectory(env, start=(5.0,5.0,0.0), goal=(35.0,35.0,0.0), N=20,
                                w_acc=1.0, w_len=0.05, w_risk=1.0, margin=margin,
                                ipopt_opts={'ipopt.print_level': 0, 'print_time': False, 'ipopt.max_iter': 200})
    X_opt = result["X_opt"]
    # For each node, sample SDF (trilinear) and assert >= margin - small_eps
    from trajectory_module import trilinear_interpolate
    eps = 1e-3
    for p in X_opt:
        d = trilinear_interpolate(sdf, p)
        assert d + eps >= margin, f"SDF violation: d={d} < margin={margin}"

def test_metrics_improve_or_reasonable(env_and_basic):
    env = env_and_basic["env"]
    # Run optimizer with moderate weights
    result = optimize_trajectory(env, start=(5.0,5.0,0.0), goal=(35.0,35.0,0.0), N=20,
                                w_acc=1.0, w_len=0.05, w_risk=6.0, margin=0.5,
                                ipopt_opts={'ipopt.print_level': 0, 'print_time': False, 'ipopt.max_iter': 200})
    metrics = result["metrics"]
    # Risk after should not be dramatically worse than before
    assert "risk_before" in metrics and "risk_after" in metrics
    assert metrics["risk_after"] <= metrics["risk_before"] + 1e-6
    # Solver info: ensure solver returned a result object with status
    solver_info = result.get("solver_info", None)
    assert solver_info is not None
    # CasADi solver result typically has 'f' and 'x'; ensure x exists
    assert hasattr(solver_info, 'x') or ('x' in solver_info), "Solver result missing solution vector"
