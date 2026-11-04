"""
=============================================================================
LINEAR ALGEBRA PROJECT - TOPIC: TWO-PLAYER ZERO-SUM MATRIX GAMES
Module: 2x2 Zero-Sum Matrix Game Solver (Algebraic & 2D Graphical Methods)
=============================================================================

For A = [[a11, a12], [a21, a22]] with NO pure saddle point, let
    D = (a11 + a22) - (a12 + a21)           (D != 0 whenever there is no saddle point)
Indifference principle:
    x1* = (a22 - a21) / D ,   x2* = 1 - x1*
    y1* = (a22 - a12) / D ,   y2* = 1 - y1*
    v   = (a11*a22 - a12*a21) / D = x*^T A y*
"""

from dataclasses import dataclass, field

import numpy as np

from .saddle_point import analyze_saddle_points


@dataclass
class Game2x2Result:
    x: np.ndarray
    y: np.ndarray
    value: float
    has_saddle_point: bool
    saddle_points: list
    maximin: float
    minimax: float
    denominator: float = None          # D, only when the game is solved in mixed strategies
    steps: list = field(default_factory=list)


def solve_2x2_detailed(A):
    """Solve a 2x2 game and return a Game2x2Result with a step-by-step explanation."""
    A = np.array(A, dtype=float)
    if A.shape != (2, 2):
        raise ValueError("Payoff matrix A must be of size 2x2.")

    a11, a12, a21, a22 = A[0, 0], A[0, 1], A[1, 0], A[1, 1]
    sp = analyze_saddle_points(A)
    steps = [
        f"Row minima = {np.array2string(sp['row_minima'], precision=4)}, "
        f"column maxima = {np.array2string(sp['col_maxima'], precision=4)}.",
        f"Maximin = {sp['maximin']:.4f}, Minimax = {sp['minimax']:.4f}.",
    ]

    if sp["has_saddle_point"]:
        i, j = sp["saddle_points"][0]
        x = np.zeros(2)
        y = np.zeros(2)
        x[i], y[j] = 1.0, 1.0
        steps.append(
            f"Maximin = Minimax -> pure saddle point at (X{i + 1}, Y{j + 1}); value v = {sp['value']:.4f}."
        )
        return Game2x2Result(x, y, float(sp["value"]), True, sp["saddle_points"],
                             sp["maximin"], sp["minimax"], None, steps)

    steps.append("Maximin < Minimax -> no pure saddle point; use the indifference principle.")
    D = (a11 + a22) - (a12 + a21)
    if np.isclose(D, 0.0):
        # Cannot happen without a saddle point; kept as a defensive guard.
        raise ZeroDivisionError("D = (a11 + a22) - (a12 + a21) = 0: the 2x2 formula does not apply.")

    x1 = (a22 - a21) / D
    y1 = (a22 - a12) / D
    v = (a11 * a22 - a12 * a21) / D
    x = np.array([x1, 1.0 - x1])
    y = np.array([y1, 1.0 - y1])

    steps += [
        f"D = (a11 + a22) - (a12 + a21) = {D:.4f}.",
        f"x* = ({x[0]:.4f}, {x[1]:.4f}),  y* = ({y[0]:.4f}, {y[1]:.4f}).",
        f"v = (a11*a22 - a12*a21) / D = {v:.4f}.",
        f"Check: Maximin ({sp['maximin']:.4f}) < v ({v:.4f}) < Minimax ({sp['minimax']:.4f}).",
    ]
    return Game2x2Result(x, y, float(v), False, [], sp["maximin"], sp["minimax"], float(D), steps)


def graphical_data(A, points=201):
    """
    Data for the graphical (envelope) method, valid when A has 2 rows and/or 2 columns.

    'row'  (m == 2): Player 1 mixes (t, 1-t) over the two rows. Each column j gives the line
                     E_j(t) = a_1j*t + a_2j*(1-t); Player 1 maximises the LOWER envelope.
    'col'  (n == 2): Player 2 mixes (s, 1-s) over the two columns. Each row i gives the line
                     F_i(s) = a_i1*s + a_i2*(1-s); Player 2 minimises the UPPER envelope.
    """
    A = np.asarray(A, dtype=float)
    m, n = A.shape
    t = np.linspace(0.0, 1.0, points)
    out = {}
    if m == 2:
        lines = np.array([A[0, j] * t + A[1, j] * (1 - t) for j in range(n)])
        env = lines.min(axis=0)
        k = int(np.argmax(env))
        out["row"] = {"t": t, "lines": lines, "envelope": env, "t_opt": float(t[k]), "v_opt": float(env[k])}
    if n == 2:
        lines = np.array([A[i, 0] * t + A[i, 1] * (1 - t) for i in range(m)])
        env = lines.max(axis=0)
        k = int(np.argmin(env))
        out["col"] = {"t": t, "lines": lines, "envelope": env, "t_opt": float(t[k]), "v_opt": float(env[k])}
    return out


def solve_game_2x2(A, plot_graph=False, save_fig_path=None, verbose=False):
    """
    Backward-compatible wrapper: returns (x_star, y_star, v).

    plot_graph / save_fig_path : draw the lower-envelope picture with matplotlib
                                 (imported lazily, only when requested).
    verbose                    : print the step-by-step explanation.
    """
    res = solve_2x2_detailed(A)
    if verbose:
        print("\n".join(res.steps))

    if plot_graph or save_fig_path:
        import matplotlib
        if save_fig_path and not plot_graph:
            matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        A = np.asarray(A, dtype=float)
        g = graphical_data(A)["row"]
        fig, ax = plt.subplots(figsize=(8, 5.5))
        for j in range(2):
            ax.plot(g["t"], g["lines"][j], linewidth=2,
                    label=f"$E_{j + 1}(x_1)$ against column {j + 1}")
        ax.plot(g["t"], g["envelope"], "r--", linewidth=2.5, label=r"Lower envelope $\min_j E_j$")
        ax.plot(g["t_opt"], g["v_opt"], "ro", markersize=8,
                label=f"Optimum: $x_1^*={g['t_opt']:.3f}$, $v={g['v_opt']:.3f}$")
        ax.set_xlabel("Probability $x_1$ (Player 1 plays row 1)")
        ax.set_ylabel("Expected payoff")
        ax.set_title("Graphical solution of a 2x2 zero-sum game")
        ax.grid(True, linestyle="--", alpha=0.6)
        ax.legend(loc="best")
        fig.tight_layout()
        if save_fig_path:
            fig.savefig(save_fig_path, dpi=200)
        if plot_graph:
            plt.show()
        plt.close(fig)

    return res.x, res.y, res.value


if __name__ == "__main__":
    x_opt, y_opt, val = solve_game_2x2([[3.0, 1.0], [0.0, 4.0]], verbose=True)
    print("Player 1 strategy x* =", x_opt)
    print("Player 2 strategy y* =", y_opt)
    print("Game value v =", val)
