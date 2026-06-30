"""
test_plant.py — guards the Phase 0 refactor.

Run from the repo root:
    python -m pytest test_plant.py -q       (or just: python test_plant.py)

Checks:
  1. The linear A, B equal the true Jacobian of the nonlinear simulator.
  2. The NumPy simulator and the symbolic model agree everywhere.
"""

import numpy as np
from plant import CartPolePlant


def _finite_diff_jacobian(deriv_fn, x, u, eps=1e-6):
    n = len(x)
    A = np.zeros((n, n))
    for i in range(n):
        dx = np.zeros(n); dx[i] = eps
        A[:, i] = (deriv_fn(x + dx, u) - deriv_fn(x - dx, u)) / (2 * eps)
    B = ((deriv_fn(x, u + eps) - deriv_fn(x, u - eps)) / (2 * eps)).reshape(-1, 1)
    return A, B


def test_AB_match_nonlinear_jacobian():
    env = CartPolePlant(1.0, 0.1, 0.5)
    env._viscous = env._coulomb = 0.0  # frictionless nominal
    A_true, B_true = _finite_diff_jacobian(env._derivatives, np.zeros(4), 0.0)
    assert np.allclose(env.A, A_true, atol=1e-4), "A is not the true Jacobian"
    assert np.allclose(env.B, B_true, atol=1e-4), "B is not the true Jacobian"


def test_simulator_matches_symbolic_model():
    env = CartPolePlant(1.0, 0.1, 0.5)
    assert env.assert_consistent()


if __name__ == "__main__":
    test_AB_match_nonlinear_jacobian()
    test_simulator_matches_symbolic_model()
    print("All Phase 0 guards pass: A,B correct and simulator == symbolic model.")