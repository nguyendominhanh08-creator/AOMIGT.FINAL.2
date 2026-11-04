"""
Preset games for the app: the three applications and two worked examples of the team report.

Each scenario documents (as required by the assignment):
  * context and modelling assumptions,
  * interpretation of the matrix elements a_ij,
  * the meaning of the optimal strategies,
  * an economic / social analysis that goes beyond the computation.
Numbers shown in the static text refer to the PRESET payoff matrix; the app also shows
insights that are recomputed from whatever matrix is currently in the editor.
"""

from dataclasses import dataclass, field

import numpy as np


@dataclass
class Scenario:
    key: str
    title: str
    kind: str                       # which solution regime the preset illustrates
    row_player: str
    col_player: str
    row_names: list
    col_names: list
    A: np.ndarray
    unit: str
    context: str
    assumptions: list
    element_meaning: str
    strategy_meaning: str
    analysis: list
    limitations: str
    references: list = field(default_factory=list)


SCENARIOS = [
    Scenario(
        key='phone', title='Application 5.1 - Smartphone advertising competition (2x2)', kind='Mixed strategies: 2x2 closed form',
        row_player='Firm A', col_player='Firm B',
        row_names=['A1: Camera and content creation', 'A2: Battery life and all-day use'], col_names=['B1: Gaming performance', 'B2: Design and personal style'],
        A=np.array([[20.0, -20.0], [-30.0, 10.0]]),
        unit='percentage points of choice share, s_A - s_B',
        context='Two hypothetical firms choose a primary advertising message for 1,000 synthetic customers.',
        assumptions=['Fixed customer groups and ranks; each customer picks the higher-ranked alternative.'],
        element_meaning='a_ij = 100 (N_A - N_B) / 1000 for message pair (Ai, Bj).',
        strategy_meaning='x* = (0.5, 0.5), y* = (0.375, 0.625), v = -5.',
        analysis='',
        limitations='All counts and ranks are assumptions of the scenario.',
    ),
    Scenario(
        key='bidding', title='Application 5.2 - Core banking project bidding (2x2)', kind='Mixed strategies: 2x2 closed form',
        row_player='Firm A', col_player='Firm B',
        row_names=['A1: Aggressive price proposal', 'A2: Technical value proposal'], col_names=['B1: Standard low-cost proposal', 'B2: Premium ecosystem proposal'],
        A=np.array([[20.0, -40.0], [0.0, 20.0]]),
        unit='percentage points of evaluation support, A minus B',
        context='Two hypothetical IT firms submit sealed proposals; five panels split 1,000 support-weight units.',
        assumptions=['Each panel supports the proposal it ranks higher.'],
        element_meaning='a_ij = support share of A minus support share of B.',
        strategy_meaning='x* = (0.25, 0.75), y* = (0.75, 0.25), v = 5.',
        analysis='',
        limitations='Weights and rankings are assumptions; not a model of a real tender.',
    ),
    Scenario(
        key='traffic', title='Application 5.3 - Traffic enforcement, adversarial model (2x2)', kind='Mixed strategies: 2x2 closed form',
        row_player='Authority', col_player='Road user (adversary)',
        row_names=['A1: Deploy enforcement', 'A2: Do not deploy'], col_names=['B1: Comply with the signal', 'B2: Violate the signal'],
        A=np.array([[-1.0, 3.5], [0.0, -50.54]]),
        unit='abstract authority score (B=0, Ce=1, Cp=4.5, Ca=50.54)',
        context='The authority chooses to deploy or not; the column player minimises the authority score.',
        assumptions=['Payoffs are abstract scores, not fines or money.'],
        element_meaning='Rows: B-Ce, Cp-Ce / B, B-Ca.',
        strategy_meaning='x* = (0.918241, 0.081759), y* = (0.981831, 0.018169), v = -0.918241.',
        analysis='',
        limitations='Zero-sum worst-case construction, not a behavioural model of drivers.',
    ),
    Scenario(
        key='ex33', title='Worked example 3.5 - Three by three game', kind='General LP: 3x3',
        row_player='Row player', col_player='Column player',
        row_names=['X1', 'X2', 'X3'], col_names=['Y1', 'Y2', 'Y3'],
        A=np.array([[3.0, -1.0, -3.0], [-2.0, 4.0, -1.0], [-5.0, -6.0, 2.0]]),
        unit='payoff units',
        context='No saddle point (L = -2 < U = 2).',
        assumptions=[],
        element_meaning='a_ij is the payoff to the row player.',
        strategy_meaning='x* = (43, 34, 32)/109, y* = (35, 9, 65)/109, v = -99/109.',
        analysis='',
        limitations='Illustrative matrix.',
    ),
    Scenario(
        key='ex34', title='Worked example - Rectangular 3x4 game', kind='General LP: 3x4',
        row_player='Row player', col_player='Column player',
        row_names=['X1', 'X2', 'X3'], col_names=['Y1', 'Y2', 'Y3', 'Y4'],
        A=np.array([[-1.0, -6.0, -5.0, -4.0], [-4.0, -1.0, -6.0, -3.0], [-5.0, -4.0, -1.0, -6.0]]),
        unit='payoff units',
        context='Rectangular game; the second column is unused in equilibrium.',
        assumptions=[],
        element_meaning='a_ij is the payoff to the row player.',
        strategy_meaning='x* = (1, 8, 5)/14, y* = (1, 0, 5, 8)/14, v = -29/7.',
        analysis='',
        limitations='Illustrative matrix.',
    ),
]


def get_scenario(key):
    return next(s for s in SCENARIOS if s.key == key)


def scenario_insights(row_names, col_names, report, unit=""):
    """Plain-language insights recomputed from the CURRENT matrix and its solution (markdown lines)."""
    A = report.A
    sp = report.saddle
    v = report.value
    lines = []

    lines.append(
        f"**Security levels.** Player 1 can guarantee at least **{sp['maximin']:.4f}** with a pure strategy (Maximin); "
        f"Player 2 can hold Player 1 to at most **{sp['minimax']:.4f}** with a pure strategy (Minimax)."
    )
    if sp["has_saddle_point"]:
        lines.append("Maximin = Minimax: a pure saddle point exists, so randomising brings no benefit to either player.")
    else:
        g1 = v - sp["maximin"]
        g2 = sp["minimax"] - v
        lines.append(
            f"**Value of randomisation.** Mixing raises Player 1's guaranteed payoff from {sp['maximin']:.4f} to "
            f"{v:.4f} (gain {g1:.4f}) and lowers Player 2's ceiling from {sp['minimax']:.4f} to {v:.4f} (gain {g2:.4f})."
        )

    def describe(names, p, who):
        used = [f"{nm} ({pr * 100:.1f}%)" for nm, pr in zip(names, p) if pr > 1e-9]
        unused = [nm for nm, pr in zip(names, p) if pr <= 1e-9]
        txt = f"**{who} plays** " + ", ".join(used) + "."
        if unused:
            txt += f" Never used in equilibrium: {', '.join(unused)}."
        return txt

    lines.append(describe(row_names, report.x, "Player 1 (rows)"))
    lines.append(describe(col_names, report.y, "Player 2 (columns)"))
    lines.append(f"**Expected payoff** x^T A y = {report.verification['xAy']:.4f}{(' ' + unit) if unit else ''}.")
    return lines
