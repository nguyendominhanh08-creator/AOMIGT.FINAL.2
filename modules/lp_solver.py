"""
Matrix game  <->  Linear programming  (Dantzig 1951; von Neumann's Minimax Theorem).

Direct formulation (A is m x n, Player 1 maximises):
    Player 1:  max v   s.t.  A^T x >= v*1,  1^T x = 1,  x >= 0
    Player 2:  min w   s.t.  A y   <= w*1,  1^T y = 1,  y >= 0
These two LPs are duals of each other, so optimal v = w (the Minimax Theorem).

Normalised formulation (used for solving). Shift B = A + k*J > 0 (k = 0 if A > 0, else 1 - min A):
    Player 2:  max 1^T q   s.t.  B q <= 1,    q >= 0           (z* = 1 / v_B)
    Player 1:  min 1^T p   s.t.  B^T p >= 1,  p >= 0           (dual, same optimum z*)
    x = p / sum(p),  y = q / sum(q),  v = 1 / z* - k
"""

from dataclasses import dataclass

import numpy as np
from scipy.optimize import linprog


@dataclass
class GameSolution:
    value: float
    x: np.ndarray
    y: np.ndarray
    formulation: str = "ferguson"
    shift: float = 0.0


def _check_matrix(A):
    A = np.asarray(A, dtype=float)
    if A.ndim != 2 or min(A.shape) == 0:
        raise ValueError("Matrix A must be a non-empty 2D array (m x n).")
    if not np.all(np.isfinite(A)):
        raise ValueError("Matrix A contains non-finite values (NaN or Inf).")
    return A


def positive_shift(A):
    """Constant k such that B = A + k > 0 entry-wise (k = 0 when A is already positive)."""
    A = np.asarray(A, dtype=float)
    return 0.0 if A.min() > 0 else 1.0 - float(A.min())


def solve_matrix_game(A, method="highs"):
    """
    Solve a general m x n zero-sum matrix game with linear programming (SciPy / HiGHS).

    Both players' LPs are solved separately, so the equality of their optimal values
    (the Minimax Theorem) is verified numerically instead of being assumed.
    """
    A_orig = _check_matrix(A)
    # Work on a magnitude-normalised copy: optimal strategies are invariant under A -> A / s,
    # and the value scales linearly. This keeps the LP well conditioned for payoffs such as 8,000,000.
    scale = float(np.max(np.abs(A_orig))) or 1.0
    A = A_orig / scale
    m, n = A.shape
    shift = positive_shift(A)
    B = A + shift

    # Player 1: min sum(p)  s.t.  B^T p >= 1, p >= 0
    r1 = linprog(np.ones(m), A_ub=-B.T, b_ub=-np.ones(n), bounds=[(0, None)] * m, method=method)
    # Player 2: max sum(q)  s.t.  B q <= 1, q >= 0
    r2 = linprog(-np.ones(n), A_ub=B, b_ub=np.ones(m), bounds=[(0, None)] * n, method=method)

    for res, who in ((r1, "Player 1"), (r2, "Player 2")):
        if not res.success:
            raise RuntimeError(f"Linear programming failed for {who}: {res.message}")

    z1, z2 = float(r1.fun), float(-r2.fun)
    if abs(z1 - z2) > 1e-7 * max(1.0, abs(z1)):
        raise RuntimeError(f"Primal/dual optima disagree ({z1} vs {z2}); the LP solution is unreliable.")

    x = np.clip(r1.x, 0, None)
    y = np.clip(r2.x, 0, None)
    return GameSolution((1.0 / z1 - shift) * scale, x / x.sum(), y / y.sum(), "ferguson", shift)


def verify_solution(A, x, y, value=None, tol=1e-7):
    """
    Check that (x, y) is an optimal pair, i.e. a saddle point of the mixed extension:
        (x^T A)_j >= v for every column j,   (A y)_i <= v for every row i,   x^T A y = v.
    """
    A = _check_matrix(A)
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    col_payoffs = x @ A            # Player 1's payoff against each pure column
    row_payoffs = A @ y            # Player 1's payoff of each pure row against y
    xAy = float(x @ A @ y)
    v = xAy if value is None else float(value)
    tol = tol * (float(np.max(np.abs(A))) or 1.0)     # tolerance relative to the size of the payoffs

    p1_guarantee = float(col_payoffs.min())
    p2_guarantee = float(row_payoffs.max())
    ok = (
        abs(x.sum() - 1) <= tol and abs(y.sum() - 1) <= tol
        and x.min() >= -tol and y.min() >= -tol
        and p1_guarantee >= v - tol
        and p2_guarantee <= v + tol
        and abs(xAy - v) <= tol
    )
    return {
        "col_payoffs": col_payoffs,
        "row_payoffs": row_payoffs,
        "xAy": xAy,
        "p1_guarantee": p1_guarantee,
        "p2_guarantee": p2_guarantee,
        "is_optimal": bool(ok),
    }


# ---------------------------------------------------------------------------
# LaTeX rendering of the LP models for the actual matrix (used by the app)
# ---------------------------------------------------------------------------
def _num(v):
    v = float(v)
    return str(int(round(v))) if abs(v - round(v)) < 1e-9 else f"{v:.4g}"


def _linear_expr(coefs, var):
    parts = []
    for k, c in enumerate(coefs, start=1):
        term = f"{_num(abs(c))}{var}_{{{k}}}"
        if not parts:
            parts.append(("-" if c < 0 else "") + term)
        else:
            parts.append((" - " if c < 0 else " + ") + term)
    return "".join(parts)


def lp_models_latex(A):
    """Return LaTeX strings describing the equivalent LP models for the matrix A."""
    A = _check_matrix(A)
    m, n = A.shape
    k = positive_shift(A)
    B = A + k

    p1_rows = [rf"{_linear_expr(A[:, j], 'x')} \ge v" for j in range(n)]
    p2_rows = [rf"{_linear_expr(A[i, :], 'y')} \le w" for i in range(m)]
    nx = " + ".join(f"x_{{{i + 1}}}" for i in range(m)) + " = 1"
    ny = " + ".join(f"y_{{{j + 1}}}" for j in range(n)) + " = 1"

    primal = (
        r"\begin{aligned}"
        r"&\max\ v \\ &\text{s.t. }"
        + r" \\ &\quad ".join(p1_rows)
        + rf" \\ &\quad {nx},\quad x_i \ge 0"
        r"\end{aligned}"
    )
    dual = (
        r"\begin{aligned}"
        r"&\min\ w \\ &\text{s.t. }"
        + r" \\ &\quad ".join(p2_rows)
        + rf" \\ &\quad {ny},\quad y_j \ge 0"
        r"\end{aligned}"
    )
    q_rows = [rf"{_linear_expr(B[i, :], 'q')} \le 1" for i in range(m)]
    p_rows = [rf"{_linear_expr(B[:, j], 'p')} \ge 1" for j in range(n)]
    obj_q = " + ".join(f"q_{{{j + 1}}}" for j in range(n))
    obj_p = " + ".join(f"p_{{{i + 1}}}" for i in range(m))

    normalised_p2 = (
        rf"\begin{{aligned}}&\max\ z = {obj_q} \\ &\text{{s.t. }}"
        + r" \\ &\quad ".join(q_rows)
        + r" \\ &\quad q_j \ge 0\end{aligned}"
    )
    normalised_p1 = (
        rf"\begin{{aligned}}&\min\ z = {obj_p} \\ &\text{{s.t. }}"
        + r" \\ &\quad ".join(p_rows)
        + r" \\ &\quad p_i \ge 0\end{aligned}"
    )
    return {
        "shift": k,
        "direct_player1": primal,
        "direct_player2": dual,
        "normalised_player2": normalised_p2,
        "normalised_player1": normalised_p1,
        "recovery": rf"x = \dfrac{{p}}{{\sum p_i}},\quad y = \dfrac{{q}}{{\sum q_j}},\quad v = \dfrac{{1}}{{z^*}} - {_num(k)}",
    }
