"""
A small, transparent tableau-simplex solver for matrix games (own implementation).

It solves Player 2's normalised LP
        max 1^T q   s.t.  B q <= 1,  q >= 0        with B = A + k > 0,
whose slack basis is feasible at q = 0.  Bland's rule (smallest index) guarantees
termination even on degenerate games.  At the optimum the reduced costs of the slack
columns are exactly the optimal dual variables p of Player 1, so one run yields
BOTH players' optimal strategies.
"""

from dataclasses import dataclass, field

import numpy as np

from .lp_solver import positive_shift


@dataclass
class SimplexResult:
    value: float
    x: np.ndarray
    y: np.ndarray
    shift: float
    iterations: list = field(default_factory=list)   # [{"basis": [...], "tableau": ndarray, "pivot": (r, c) | None}]
    labels: list = field(default_factory=list)       # column labels of the tableau


def simplex_solve(A, tol=1e-10, max_iter=10_000):
    A = np.asarray(A, dtype=float)
    if A.ndim != 2 or A.size == 0 or not np.all(np.isfinite(A)):
        raise ValueError("Matrix A must be a finite, non-empty 2D array.")
    scale = float(np.max(np.abs(A))) or 1.0      # normalise magnitude; strategies are scale invariant
    A = A / scale
    m, n = A.shape
    k = positive_shift(A)
    B = A + k

    # Tableau: rows 0..m-1 constraints, last row = objective (z - sum q = 0).
    T = np.zeros((m + 1, n + m + 1))
    T[:m, :n] = B
    T[:m, n:n + m] = np.eye(m)
    T[:m, -1] = 1.0
    T[m, :n] = -1.0
    basis = list(range(n, n + m))
    labels = [f"q{j + 1}" for j in range(n)] + [f"s{i + 1}" for i in range(m)] + ["RHS"]

    iterations = [{"basis": [labels[b] for b in basis], "tableau": T.copy(), "pivot": None}]

    for _ in range(max_iter):
        negatives = np.where(T[m, :-1] < -tol)[0]
        if negatives.size == 0:
            break
        col = int(negatives[0])                              # Bland: smallest index entering
        positive = np.where(T[:m, col] > tol)[0]
        if positive.size == 0:
            raise RuntimeError("LP unbounded - impossible for B > 0; check the input matrix.")
        ratios = T[positive, -1] / T[positive, col]
        best = ratios.min()
        ties = positive[np.isclose(ratios, best, atol=tol)]
        row = int(min(ties, key=lambda r: basis[r]))         # Bland: smallest leaving variable
        T[row] /= T[row, col]
        for r in range(m + 1):
            if r != row:
                T[r] -= T[r, col] * T[row]
        basis[row] = col
        iterations.append({"basis": [labels[b] for b in basis], "tableau": T.copy(), "pivot": (row, col)})
    else:
        raise RuntimeError("Simplex did not converge within the iteration limit.")

    q = np.zeros(n)
    for r, b in enumerate(basis):
        if b < n:
            q[b] = T[r, -1]
    p = np.clip(T[m, n:n + m], 0, None)          # dual values = Player 1's variables
    z = float(T[m, -1])

    return SimplexResult(
        value=(1.0 / z - k) * scale,
        x=p / p.sum(),
        y=q / q.sum(),
        shift=k,
        iterations=iterations,
        labels=labels,
    )
