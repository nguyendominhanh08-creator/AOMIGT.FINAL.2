"""
Iterated elimination of strictly dominated pure strategies.

* Row i is strictly dominated by row k  if  a_kj > a_ij  for every column j
  (Player 1 maximises, so a dominated row is never used in an optimal strategy).
* Column j is strictly dominated by column l  if  a_il < a_ij  for every row i
  (Player 2 minimises, so the column with larger payoffs is dominated).

Removing strictly dominated strategies does not change the value of the game or
the set of optimal strategies, which makes it a cheap pre-processing step
before the general linear-programming algorithm.
"""

import numpy as np


def eliminate_dominated(A, tol=1e-12):
    """
    Repeatedly delete strictly dominated rows and columns.

    Returns a dict with
        reduced   : np.ndarray   the reduced payoff matrix
        row_idx   : list[int]    surviving original row indices (0-based)
        col_idx   : list[int]    surviving original column indices (0-based)
        steps     : list[str]    human-readable log (1-based labels)
    """
    A = np.asarray(A, dtype=float)
    rows = list(range(A.shape[0]))
    cols = list(range(A.shape[1]))
    steps = []

    changed = True
    while changed:
        changed = False
        sub = A[np.ix_(rows, cols)]

        # Rows (maximiser): drop row i if some other row k is strictly better everywhere.
        for a in range(len(rows)):
            dominator = next(
                (b for b in range(len(rows)) if b != a and np.all(sub[b, :] > sub[a, :] + tol)),
                None,
            )
            if dominator is not None and len(rows) > 1:
                steps.append(
                    f"Row X{rows[a] + 1} is strictly dominated by row X{rows[dominator] + 1} -> removed."
                )
                rows.pop(a)
                changed = True
                break
        if changed:
            continue

        # Columns (minimiser): drop column j if some other column l is strictly smaller everywhere.
        for a in range(len(cols)):
            dominator = next(
                (b for b in range(len(cols)) if b != a and np.all(sub[:, b] < sub[:, a] - tol)),
                None,
            )
            if dominator is not None and len(cols) > 1:
                steps.append(
                    f"Column Y{cols[a] + 1} is strictly dominated by column Y{cols[dominator] + 1} -> removed."
                )
                cols.pop(a)
                changed = True
                break

    if not steps:
        steps.append("No strictly dominated pure strategy found; the matrix cannot be reduced.")

    return {
        "reduced": A[np.ix_(rows, cols)],
        "row_idx": rows,
        "col_idx": cols,
        "steps": steps,
    }


def lift_strategy(p_reduced, kept_idx, size):
    """Embed a strategy of the reduced game into the original strategy space."""
    p = np.zeros(size, dtype=float)
    p[list(kept_idx)] = np.asarray(p_reduced, dtype=float)
    return p
