"""
Command-line demo (no Streamlit needed):  python main.py

Solves the three real-world scenarios and prints the full solution report.
For the web interface run:  streamlit run app.py
"""

import numpy as np

from modules.scenarios import SCENARIOS, scenario_insights
from modules.solver import solve_game


def show(scenario):
    rep = solve_game(scenario.A)
    print("=" * 78)
    print(scenario.title)
    print("=" * 78)
    print("Payoff matrix A (rows = %s, columns = %s):" % (scenario.row_player, scenario.col_player))
    print(np.array2string(scenario.A, precision=2))
    for line in rep.steps:
        print(" -", line)
    print("Method:", rep.method)
    print("x* =", np.round(rep.x, 4))
    print("y* =", np.round(rep.y, 4))
    print("v  = %.4f   (x*^T A y* = %.4f)" % (rep.value, rep.verification["xAy"]))
    print("Optimal pair verified:", rep.verification["is_optimal"],
          "| HiGHS and simplex agree:", rep.values_agree)
    print()
    for line in scenario_insights(scenario.row_names, scenario.col_names, rep):
        print("  *", line.replace("**", ""))
    print()


if __name__ == "__main__":
    for sc in SCENARIOS:
        show(sc)
