# AoMiGT.TEAM2 — Applications of Matrices in Game Theory

Two-player **zero-sum matrix games**: payoff matrix `A = (a_ij)`, pure and mixed strategies, saddle points,
the Minimax Theorem, the equivalent linear-programming model, solution algorithms and real-world scenarios.

## Run

```bash
pip install -r requirements.txt
streamlit run app.py        # web interface (any m x n matrix: input + computation only)
python main.py              # terminal demo of the three scenarios
python tests/test_solver.py # automated checks (or: python -m pytest tests -q)
```

## Structure

| File | Content |
|---|---|
| `modules/saddle_point.py` | Maximin / Minimax, all pure saddle points |
| `modules/dominance.py` | Iterated elimination of strictly dominated strategies |
| `modules/solve_2x2.py` | 2x2 closed form (indifference principle) + graphical envelope method |
| `modules/lp_solver.py` | LP model of a matrix game (HiGHS), LaTeX of the LP models, optimality check |
| `modules/simplex.py` | Own tableau-simplex implementation (Bland's rule), returns every tableau |
| `modules/solver.py` | Full algorithm: saddle test -> 2x2 / dominance + LP -> verification -> cross-check |
| `modules/scenarios.py` | 5 preset matrices (3 applications + 2 worked examples) (presets). The written analysis is kept here for the report, not shown in the app |
| `app.py` | Streamlit UI: payoff matrix input and computation of the solution |

## Scenarios

1. Smartphone advertising, 2x2 - x* = (0.5, 0.5), v = -5.
2. Core banking bidding, 2x2 - x* = (0.25, 0.75), v = 5.
3. Traffic enforcement (adversarial), 2x2 - v = -0.918241.
4. Worked 3x3 example - v = -99/109.
5. Worked 3x4 example - v = -29/7.

## References

Used for the report (Osborne 2004; Taha 2017; von Neumann 1928; Dantzig 1951; Ferguson 2014;
Owen 2013; Luenberger & Ye 2016; Gordon & Loeb 2002; Tambe 2011).
