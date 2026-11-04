"""
Orchestrator: runs the complete solution algorithm for a two-player zero-sum matrix game.

Algorithm
  1. Maximin / Minimax analysis  ->  pure saddle point?  (v_lower == v_upper)
  2. If yes: pure optimal strategies, value v = a_ij.
  3. Else, 2x2 game: closed-form indifference principle.
  4. Else (general m x n): delete strictly dominated strategies, then solve the equivalent
     linear programme (HiGHS) on the reduced game and lift the strategies back.
  5. Always: verify optimality (x^T A >= v, A y <= v, x^T A y = v) and cross-check the value
     with both an LP solver (HiGHS) and our own simplex implementation.
"""

from dataclasses import dataclass, field

import numpy as np

from .dominance import eliminate_dominated, lift_strategy
from .lp_solver import solve_matrix_game, verify_solution
from .saddle_point import analyze_saddle_points
from .simplex import simplex_solve
from .solve_2x2 import solve_2x2_detailed


@dataclass
class GameReport:
    A: np.ndarray
    saddle: dict
    dominance: dict
    method: str
    x: np.ndarray
    y: np.ndarray
    value: float
    verification: dict
    lp: object
    lp_verification: dict
    simplex: object
    simplex_verification: dict
    values_agree: bool
    two_by_two: object = None
    steps: list = field(default_factory=list)


def solve_game(A, tol=1e-6):
    A = np.asarray(A, dtype=float)
    if A.ndim != 2 or A.size == 0 or not np.all(np.isfinite(A)):
        raise ValueError("Payoff matrix must be a finite, non-empty 2D array.")
    m, n = A.shape

    saddle = analyze_saddle_points(A)
    dominance = eliminate_dominated(A)
    steps = [
        f"Maximin = {saddle['maximin']:.4f}, Minimax = {saddle['minimax']:.4f} "
        f"(gap = {saddle['gap']:.4f}).",
    ]
    two_by_two = None

    if saddle["has_saddle_point"]:
        i, j = saddle["saddle_points"][0]
        x = np.zeros(m)
        y = np.zeros(n)
        x[i], y[j] = 1.0, 1.0
        value = float(A[i, j])
        method = "Pure strategies (saddle point)"
        steps.append(f"Maximin = Minimax -> saddle point at (X{i + 1}, Y{j + 1}), v = {value:.4f}.")
    elif (m, n) == (2, 2):
        two_by_two = solve_2x2_detailed(A)
        x, y, value = two_by_two.x, two_by_two.y, two_by_two.value
        method = "2x2 closed form (indifference principle)"
        steps.append("Maximin < Minimax -> mixed strategies via the 2x2 formulas.")
    else:
        red = dominance["reduced"]
        steps.append("Maximin < Minimax -> no pure saddle point; solve the equivalent linear programme.")
        if red.shape != A.shape:
            steps.append(
                f"Dominance reduced the game from {m}x{n} to {red.shape[0]}x{red.shape[1]} before the LP."
            )
        sol = solve_matrix_game(red)
        x = lift_strategy(sol.x, dominance["row_idx"], m)
        y = lift_strategy(sol.y, dominance["col_idx"], n)
        value = float(sol.value)
        method = ("Linear programming (HiGHS) after dominance reduction"
                  if red.shape != A.shape else "Linear programming (HiGHS)")

    verification = verify_solution(A, x, y, value, tol)

    lp = solve_matrix_game(A)
    sx = simplex_solve(A)
    lp_ver = verify_solution(A, lp.x, lp.y, lp.value, tol)
    sx_ver = verify_solution(A, sx.x, sx.y, sx.value, tol)
    tol_v = tol * (float(np.max(np.abs(A))) or 1.0)          # relative to the size of the payoffs
    agree = abs(value - lp.value) <= tol_v and abs(value - sx.value) <= tol_v

    return GameReport(
        A=A, saddle=saddle, dominance=dominance, method=method, x=x, y=y, value=value,
        verification=verification, lp=lp, lp_verification=lp_ver, simplex=sx,
        simplex_verification=sx_ver, values_agree=agree, two_by_two=two_by_two, steps=steps,
    )
