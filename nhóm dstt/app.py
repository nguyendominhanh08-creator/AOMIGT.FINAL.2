import streamlit as st
import numpy as np
import pandas as pd
from modules.saddle_point import find_saddle_points
from modules.solve_2x2 import solve_game_2x2
from modules.lp_solver import solve_matrix_game

# Page Configuration
st.set_page_config(page_title="Game Theory Solver App", layout="wide")

st.title("🎮 GAME THEORY SOLVER")
st.caption("Interactive Application for Zero-Sum Matrix Games - Linear Algebra Project")

st.sidebar.header("⚙️ Matrix Configuration")
rows = st.sidebar.number_input("Number of Rows (Player 1 Strategies):", min_value=2, max_value=6, value=2, step=1)
cols = st.sidebar.number_input("Number of Columns (Player 2 Strategies):", min_value=2, max_value=6, value=2, step=1)

st.subheader("1. Enter Payoff Matrix A:")

# Create default float matrix formatted as pandas DataFrame to allow smooth grid editing
default_matrix = pd.DataFrame(
    np.zeros((rows, cols), dtype=float),
    columns=[f"Col {j+1}" for j in range(cols)],
    index=[f"Row {i+1}" for i in range(rows)]
)

# Render editable table
edited_df = st.data_editor(
    default_matrix,
    key=f"editor_{rows}_{cols}",
    num_rows="fixed",
    use_container_width=True
)

if st.button("🚀 Solve Game", type="primary"):
    # Convert DataFrame back to 2D numpy array float
    A = edited_df.to_numpy(dtype=float)
    
    st.markdown("---")
    st.subheader("2. Solution & Analysis Results:")
    
    # Step 1: Check Pure Strategy (Saddle Point - TV1)
    try:
        sp_result = find_saddle_points(A)
        has_sp = False
        saddle_pts = []
        
        # Sửa điều kiện nhận diện linh hoạt cho cả dict lẫn list
        if isinstance(sp_result, dict) and sp_result.get("has_saddle_point"):
            has_sp = True
            saddle_pts = sp_result.get("saddle_points")
        elif isinstance(sp_result, list) and len(sp_result) > 0:
            has_sp = True
            saddle_pts = sp_result

        if has_sp:
            st.success("✅ Pure Strategy Equilibrium (Saddle Point) Found!")
            st.write("**Saddle Point Location(s) (row, col):**", saddle_pts)
            # Lấy vị trí dòng và cột của điểm yên ngựa đầu tiên
            r_idx, c_idx = saddle_pts[0]
            st.write(f"**Game Value (v):** `{A[r_idx, c_idx]:.6f}`")
            st.stop()
    except Exception as e:
        pass
    
    # Step 2: 2x2 Matrix Solver (TV2)
    if rows == 2 and cols == 2:
        try:
            x_opt, y_opt, val = solve_game_2x2(A)
            st.success("🎯 Mixed Strategy Equilibrium (2x2 Analytical Module):")
            st.write("**Optimal Strategy for Player 1 (x):**", np.round(x_opt, 4))
            st.write("**Optimal Strategy for Player 2 (y):**", np.round(y_opt, 4))
            st.write(f"**Game Value (v):** `{val:.6f}`")
            st.stop()
        except Exception as e:
            st.info("Direct 2x2 solver skipped, resolving with general LP Solver...")

    # Step 3: General m x n Linear Programming Solver (TV3)
    try:
        sol = solve_matrix_game(A)
        st.success("🚀 Mixed Strategy Equilibrium (General m x n LP Solver):")
        st.write("**Optimal Strategy for Player 1 (x):**", np.round(sol.x, 4))
        st.write("**Optimal Strategy for Player 2 (y):**", np.round(sol.y, 4))
        st.write(f"**Game Value (v):** `{sol.value:.6f}`")
    except Exception as e:
        st.error(f"❌ LP Solver Error: {e}")