import numpy as np


def validate_matrix(matrix_input):
    """
    Validates and formats the input payoff matrix.

    Returns (is_valid, message, matrix_or_None).
    """
    try:
        matrix = np.array(matrix_input, dtype=float)

        if matrix.ndim != 2:
            return False, "Matrix must be 2-dimensional!", None

        if matrix.size == 0 or matrix.shape[0] == 0 or matrix.shape[1] == 0:
            return False, "Matrix cannot be empty!", None

        if not np.all(np.isfinite(matrix)):
            return False, "Matrix contains empty, NaN or infinite cells. Please fill every cell with a number.", None

        return True, "Data is valid.", matrix

    except Exception as e:
        return False, f"Data format error: {str(e)}", None
