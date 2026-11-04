"""
=============================================================================
LINEAR ALGEBRA PROJECT - TOPIC: TWO-PLAYER ZERO-SUM MATRIX GAMES
Module: 2x2 Zero-Sum Matrix Game Solver (Algebraic & 2D Graphical Methods)
Language: Python 3
=============================================================================
"""

import numpy as np
import matplotlib.pyplot as plt


def solve_game_2x2(A, plot_graph=True, save_fig_path=None):
    """
    Solves a 2x2 two-player zero-sum matrix game.

    Parameters:
        A (array-like): 2x2 payoff matrix for the Row Player (Player 1).
        plot_graph (bool): If True, plots the 2D graphical solution.
        save_fig_path (str, optional): File path to save the generated plot.

    Returns:
        tuple: (x_star, y_star, v)
            - x_star (np.ndarray): Optimal mixed strategy vector for Player 1 [x1*, x2*]^T
            - y_star (np.ndarray): Optimal mixed strategy vector for Player 2 [y1*, y2*]^T
            - v (float): Value of the game
    """
    A = np.array(A, dtype=float)
    if A.shape != (2, 2):
        raise ValueError("Payoff matrix A must be of size 2x2.")

    a11, a12 = A[0, 0], A[0, 1]
    a21, a22 = A[1, 0], A[1, 1]

    print("=" * 65)
    print("PAYOFF MATRIX A (Player 1 Payoffs):")
    print(A)
    print("=" * 65)

    # -------------------------------------------------------------------------
    # STEP 1: PURE SADDLE POINT VERIFICATION (MAXIMIN / MINIMAX PRINCIPLE)
    # -------------------------------------------------------------------------
    row_mins = np.min(A, axis=1)  # Minimum payoff of each row
    col_maxs = np.max(A, axis=0)  # Maximum payoff of each column

    maximin = np.max(row_mins)    # Lower value of the game: v_lower
    minimax = np.min(col_maxs)    # Upper value of the game: v_upper

    print(f"Row Minimums (min_j a_ij): {row_mins}")
    print(f"Column Maximums (max_i a_ij): {col_maxs}")
    print(f"Maximin Value (v_lower) = {maximin:.4f} | Minimax Value (v_upper) = {minimax:.4f}")

    # Check if a pure saddle point exists: v_lower == v_upper
    if np.isclose(maximin, minimax):
        v = float(maximin)
        saddle_row = int(np.where(row_mins == maximin)[0][0])
        saddle_col = int(np.where(col_maxs == minimax)[0][0])

        x_star = np.zeros(2)
        x_star[saddle_row] = 1.0

        y_star = np.zeros(2)
        y_star[saddle_col] = 1.0

        print("\n[RESULT] A PURE STRATEGY SADDLE POINT exists.")
        print(f"  - Saddle Point Location: Row {saddle_row + 1}, Column {saddle_col + 1}")
        print(f"  - Player 1 Optimal Pure Strategy: x* = [{x_star[0]:.4f}, {x_star[1]:.4f}]^T")
        print(f"  - Player 2 Optimal Pure Strategy: y* = [{y_star[0]:.4f}, {y_star[1]:.4f}]^T")
        print(f"  - Equilibrium Game Value v = {v:.4f}")
        return x_star, y_star, v

    print("\n[RESULT] No pure saddle point found (Maximin < Minimax).")
    print("         Proceeding to compute MIXED STRATEGY EQUILIBRIUM...")

    # -------------------------------------------------------------------------
    # STEP 2: ALGEBRAIC METHOD (INDIFFERENCE PRINCIPLE)
    # -------------------------------------------------------------------------
    denominator = (a11 + a22) - (a12 + a21)

    # Edge case: Avoid division by zero
    if np.isclose(denominator, 0.0):
        raise ZeroDivisionError(
            "Denominator D = (a11 + a22) - (a12 + a21) is 0. "
            "The matrix does not admit a strictly interior mixed strategy via standard algebraic formula."
        )

    # Compute optimal probabilities using indifference equations
    x1 = (a22 - a21) / denominator
    x2 = 1.0 - x1

    y1 = (a22 - a12) / denominator
    y2 = 1.0 - y1

    # Equilibrium game value
    v = (a11 * a22 - a12 * a21) / denominator

    # Validate probability bounds [0, 1]
    if not (0.0 <= x1 <= 1.0 and 0.0 <= y1 <= 1.0):
        print("[WARNING] Calculated probabilities lie outside [0, 1]. Please verify matrix structure.")

    x_star = np.array([x1, x2])
    y_star = np.array([y1, y2])

    print("\n--- MIXED STRATEGY EQUILIBRIUM (ALGEBRAIC SOLUTION) ---")
    print(f"  - Denominator D = {denominator:.4f}")
    print(f"  - Player 1 Probabilities: p* = {x1:.4f}, (1 - p*) = {x2:.4f}")
    print(f"    => Optimal Strategy Vector: x* = [{x1:.4f}, {x2:.4f}]^T")
    print(f"  - Player 2 Probabilities: q* = {y1:.4f}, (1 - q*) = {y2:.4f}")
    print(f"    => Optimal Strategy Vector: y* = [{y1:.4f}, {y2:.4f}]^T")
    print(f"  - Value of the Game: v = {v:.4f}")
    print(f"  - Minimax Condition Check: Maximin ({maximin:.4f}) < v ({v:.4f}) < Minimax ({minimax:.4f})")

    # -------------------------------------------------------------------------
    # STEP 3: 2D GRAPHICAL METHOD (LOWER ENVELOPE)
    # -------------------------------------------------------------------------
    if plot_graph:
        x_vals = np.linspace(0, 1, 300)
        # Expected payoffs against Player 2's pure strategies
        E1 = a11 * x_vals + a21 * (1 - x_vals)  # Against Column 1
        E2 = a12 * x_vals + a22 * (1 - x_vals)  # Against Column 2

        # Lower envelope: E(x) = min(E1, E2)
        lower_envelope = np.minimum(E1, E2)

        fig, ax = plt.subplots(figsize=(8, 5.5))
        ax.plot(x_vals, E1, 'b-', linewidth=2, label=f'$E_1(x_1)$: {a11:.1f}$x_1$ + {a21:.1f}(1 - $x_1$) (against Col 1)')
        ax.plot(x_vals, E2, 'g-', linewidth=2, label=f'$E_2(x_1)$: {a12:.1f}$x_1$ + {a22:.1f}(1 - $x_1$) (against Col 2)')
        ax.plot(x_vals, lower_envelope, 'r--', linewidth=2.5, label='Lower Envelope: $\\min(E_1, E_2)$')

        # Highlight Maximin Equilibrium Point
        ax.plot(x1, v, 'ro', markersize=8, label=f'Maximin Equilibrium ($x_1^*={x1:.3f}, v={v:.3f}$)')
        ax.vlines(x=x1, ymin=min(np.min(E1), np.min(E2)), ymax=v, colors='gray', linestyles=':')
        ax.hlines(y=v, xmin=0, xmax=x1, colors='gray', linestyles=':')

        ax.set_title('2D Graphical Solution for 2x2 Zero-Sum Matrix Game', fontsize=12, fontweight='bold')
        ax.set_xlabel('Probability $x_1$ (Player 1 selects Pure Strategy 1)', fontsize=11)
        ax.set_ylabel('Expected Payoff $E(x_1)$', fontsize=11)
        ax.set_xlim([0, 1])
        ax.grid(True, linestyle='--', alpha=0.6)
        ax.legend(loc='best', frameon=True)
        fig.tight_layout()

        if save_fig_path:
            fig.savefig(save_fig_path, dpi=200)
            print(f"  -> Graphical plot saved to: {save_fig_path}")

        plt.close(fig)

    return x_star, y_star, v


# =============================================================================
# DEMONSTRATION / VERIFICATION EXAMPLE
# =============================================================================
if __name__ == '__main__':
    A_sample = np.array([
        [3.0, 1.0],
        [0.0, 4.0]
    ])
    
    # Call the solver function
    x_opt, y_opt, val = solve_game_2x2(A_sample)
    
    # Print the output to Terminal
    print("--- 2x2 Game Solution ---")
    print("Player 1 Strategy (x):", x_opt)
    print("Player 2 Strategy (y):", y_opt)
    print("Game Value (v):", val)
