"""
Saddle-point analysis for a two-player zero-sum matrix game.

Player 1 (row player, maximiser) picks a row i, Player 2 (column player,
minimiser) picks a column j, and Player 1 receives a_ij.

    lower value  v_lower = max_i min_j a_ij   (Maximin, Player 1's security level)
    upper value  v_upper = min_j max_i a_ij   (Minimax, Player 2's security level)

Always v_lower <= v_upper.  A pure saddle point exists  <=>  v_lower == v_upper,
and a saddle point is an entry a_ij that is simultaneously the minimum of its
row and the maximum of its column.
"""

import numpy as np


def _as_matrix(A):
    A = np.asarray(A, dtype=float)
    if A.ndim != 2 or A.size == 0:
        raise ValueError("Payoff matrix A must be a non-empty 2D array (m x n).")
    if not np.all(np.isfinite(A)):
        raise ValueError("Payoff matrix A contains NaN or infinite values.")
    return A


def analyze_saddle_points(A, tol=1e-9):
    """
    Full Maximin / Minimax analysis of the payoff matrix A.

    Returns a dict with
        row_minima, col_maxima      : np.ndarray
        maximin, minimax            : float   (lower / upper value of the game)
        maximin_rows, minimax_cols  : list[int]  rows/cols attaining them
        gap                         : float   minimax - maximin (>= 0)
        has_saddle_point            : bool
        saddle_points               : list[(i, j)]  0-based indices
        value                       : float | None  (value of the game if a saddle point exists)
    """
    A = _as_matrix(A)
    m, n = A.shape
    eps = tol * max(1.0, float(np.max(np.abs(A))))

    row_minima = A.min(axis=1)
    col_maxima = A.max(axis=0)
    maximin = float(row_minima.max())
    minimax = float(col_maxima.min())

    saddle_points = [
        (i, j)
        for i in range(m)
        for j in range(n)
        if abs(A[i, j] - row_minima[i]) <= eps and abs(A[i, j] - col_maxima[j]) <= eps
    ]

    return {
        "row_minima": row_minima,
        "col_maxima": col_maxima,
        "maximin": maximin,
        "minimax": minimax,
        "maximin_rows": [int(i) for i in np.where(np.abs(row_minima - maximin) <= eps)[0]],
        "minimax_cols": [int(j) for j in np.where(np.abs(col_maxima - minimax) <= eps)[0]],
        "gap": minimax - maximin,
        "has_saddle_point": len(saddle_points) > 0,
        "saddle_points": saddle_points,
        "value": maximin if saddle_points else None,
    }


def find_saddle_points(A):
    """Return all pure saddle points of A as a list of 0-based (row, col) tuples."""
    return analyze_saddle_points(A)["saddle_points"]
