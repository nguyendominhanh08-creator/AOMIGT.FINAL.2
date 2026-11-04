"""Run with:  python -m pytest tests -q   (or)   python tests/test_solver.py"""
import os
import sys

import numpy as np
from scipy.optimize import linprog

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modules.dominance import eliminate_dominated
from modules.lp_solver import solve_matrix_game, verify_solution
from modules.saddle_point import analyze_saddle_points, find_saddle_points
from modules.scenarios import SCENARIOS
from modules.simplex import simplex_solve
from modules.solve_2x2 import solve_game_2x2
from modules.solver import solve_game


def reference_value(A):
    """Independent LP with a free variable v (direct formulation)."""
    m, n = A.shape
    c = np.zeros(m + 1)
    c[-1] = -1
    res = linprog(c, A_ub=np.hstack([-A.T, np.ones((n, 1))]), b_ub=np.zeros(n),
                  A_eq=np.hstack([np.ones((1, m)), [[0]]]), b_eq=[1],
                  bounds=[(0, None)] * m + [(None, None)], method="highs")
    return res.x[-1]


def test_known_2x2_example():
    x, y, v = solve_game_2x2([[3, 1], [0, 4]])
    assert np.allclose(x, [2 / 3, 1 / 3]) and np.allclose(y, [0.5, 0.5]) and np.isclose(v, 2.0)


def test_saddle_point():
    assert find_saddle_points(np.array([[3, 2, 4], [1, 0, 5], [2, -1, 3]])) == [(0, 1)]
    assert find_saddle_points(np.array([[3, 1], [0, 4]])) == []


def test_scenarios_expected_solutions():
    ph, bd, tr, e33, e34 = [solve_game(s.A) for s in SCENARIOS]
    assert np.allclose(ph.x, [0.5, 0.5]) and np.allclose(ph.y, [0.375, 0.625]) and np.isclose(ph.value, -5)
    assert np.allclose(bd.x, [0.25, 0.75]) and np.allclose(bd.y, [0.75, 0.25]) and np.isclose(bd.value, 5)
    assert np.isclose(tr.value, -50.54 / 55.04) and np.isclose(tr.x[0], 50.54 / 55.04)
    assert np.allclose(e33.x, np.array([43, 34, 32]) / 109) and np.isclose(e33.value, -99 / 109)
    assert np.allclose(e34.x, np.array([1, 8, 5]) / 14) and np.isclose(e34.value, -29 / 7)
    for rep in (ph, bd, tr, e33, e34):
        assert rep.verification["is_optimal"] and rep.values_agree


def test_random_games_lp_simplex_dominance():
    rng = np.random.default_rng(1)
    shapes = [(2, 2), (2, 3), (3, 2), (3, 3), (3, 4), (4, 4), (5, 3), (6, 6)]
    for t in range(240):
        m, n = shapes[t % len(shapes)]
        A = rng.integers(-5, 6, (m, n)).astype(float) if t % 2 else rng.integers(0, 3, (m, n)).astype(float)
        v = reference_value(A)
        for sol in (solve_matrix_game(A), simplex_solve(A)):
            assert abs(sol.value - v) < 1e-7
            assert verify_solution(A, sol.x, sol.y, sol.value)["is_optimal"]
        assert abs(reference_value(eliminate_dominated(A)["reduced"]) - v) < 1e-7
        sp = analyze_saddle_points(A)
        assert sp["has_saddle_point"] == bool(abs(sp["maximin"] - sp["minimax"]) < 1e-9)
        rep = solve_game(A)
        assert rep.verification["is_optimal"] and rep.values_agree


def test_large_and_small_payoffs():
    base = np.array([[-1, -6, -5, -4], [-4, -1, -6, -3], [-5, -4, -1, -6]], float)
    rng = np.random.default_rng(11)
    randoms = [np.round(rng.normal(0, 5, (4, 4)), 2) for _ in range(40)]
    for scale in (1e-6, 1e3, 1e6, 1e9, 1e12):
        for A in [base] + randoms:
            rep = solve_game(A * scale)
            assert rep.verification["is_optimal"] and rep.values_agree, (scale, A.shape)
        assert np.isclose(solve_game(base * scale).value, -29 / 7 * scale, rtol=1e-9)


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("PASSED", name)
