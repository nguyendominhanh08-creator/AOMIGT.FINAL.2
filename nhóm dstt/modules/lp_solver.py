from dataclasses import dataclass
import numpy as np
from scipy.optimize import linprog


@dataclass
class GameSolution:
    value: float          
    x: np.ndarray         
    y: np.ndarray         
    formulation: str


def _check_matrix(A):
    A = np.asarray(A, dtype=float)
    if A.ndim != 2 or min(A.shape) == 0:
        raise ValueError("Matrix A must be a non-empty 2D array (m x n).")
    if not np.all(np.isfinite(A)):
        raise ValueError("Matrix A contains non-finite values (NaN or Inf).")
    return A


def _solve_ferguson(A, method):
    m, n = A.shape
    shift = 0.0 if A.min() > 0 else 1.0 - A.min()  
    B = A + shift

    r1 = linprog(c=np.ones(m), A_ub=-B.T, b_ub=-np.ones(n),
                 bounds=[(0, None)] * m, method=method)
    
    r2 = linprog(c=-np.ones(n), A_ub=B, b_ub=np.ones(m),
                 bounds=[(0, None)] * n, method=method)
    for r, who in ((r1, "Player I"), (r2, "Player II")):
        if not r.success:
            raise RuntimeError(f"linprog optimization failed for {who}: {r.message}")

    v_B = 1.0 / r1.fun                
    x = r1.x * v_B
    y = r2.x * v_B
    return v_B - shift, x, y


def _solve_direct_row(A, method):
    """Calculates optimal strategy and game value for the row player (maximizer) of matrix A."""
    m, n = A.shape
    c = np.zeros(m + 1)
    c[-1] = -1.0                                      
    A_ub = np.hstack([-A.T, np.ones((n, 1))])           
    b_ub = np.zeros(n)
    A_eq = np.hstack([np.ones((1, m)), np.zeros((1, 1))])  
    b_eq = np.array([1.0])
    bounds = [(0, None)] * m + [(None, None)]          
    r = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq,
                bounds=bounds, method=method)
    if not r.success:
        raise RuntimeError(f"linprog optimization failed: {r.message}")
    return r.x[-1], r.x[:m]


def _solve_direct(A, method):
    v, x = _solve_direct_row(A, method)
    v2, y = _solve_direct_row(-A.T, method)
    return v, x, y


def solve_matrix_game(A, formulation="ferguson", method="highs", tol=1e-8):
    """
    Solves matrix game A (m x n). Returns GameSolution(value, x, y, formulation).
    Verification check: min_j (x^T A)_j >= v - tol and max_i (A y)_i <= v + tol.
    """
    A = _check_matrix(A)
    if formulation == "ferguson":
        v, x, y = _solve_ferguson(A, method)
    elif formulation == "direct":
        v, x, y = _solve_direct(A, method)
    else:
        raise ValueError("Formulation must be either 'ferguson' or 'direct'.")

    x = np.clip(x, 0, None); x /= x.sum()
    y = np.clip(y, 0, None); y /= y.sum()

    lower = (x @ A).min()     
    upper = (A @ y).max()     
    if lower < v - 1e-6 or upper > v + 1e-6 or abs(lower - upper) > 1e-6:
        raise RuntimeError(
            f"Minimax verification failed: lower={lower:.10f}, upper={upper:.10f}, v={v:.10f}"
        )
    return GameSolution(float(v), x, y, formulation)


def find_saddle_points(A):
    """Finds pure strategy saddle points (i, j): a_ij is minimum in row i and maximum in column j."""
    A = _check_matrix(A)
    row_min = A.min(axis=1, keepdims=True)
    col_max = A.max(axis=0, keepdims=True)
    return [(int(i), int(j)) for i, j in zip(*np.where((A == row_min) & (A == col_max)))]


if __name__ == "__main__":
    np.set_printoptions(precision=6, suppress=True)
    tests = {
        "Rock-Paper-Scissors": [[0, -1, 1], [1, 0, -1], [-1, 1, 0]],
        "With Saddle Point": [[3, 1, 4], [5, 2, 6], [0, 1, 2]],
        "2x2 Matrix": [[2, -1], [-1, 1]],
        "3x4 Matrix": [[2, -1, 6, 0], [0, 1, -1, 3], [-2, 2, 1, -1]],
    }
    for name, A in tests.items():
        print(f"=== Test Case: {name} ===")
        print("Saddle Points (indices from 0):", find_saddle_points(A))
        for f in ("ferguson", "direct"):
            s = solve_matrix_game(A, formulation=f)
            print(f"[{f:8s}] v = {s.value:.6f} | x = {s.x} | y = {s.y}")
        print()