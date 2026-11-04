import numpy as np

def validate_matrix(matrix_input):
    """
    Validates and formats the input payoff matrix.
    """
    try:
        # Convert input data to 2D NumPy array with float type
        matrix = np.array(matrix_input, dtype=float)
        
        # Check if the array is 2-dimensional
        if matrix.ndim != 2:
            return False, "Matrix must be 2-dimensional!", None
            
        # Check if the matrix is empty
        if matrix.size == 0 or matrix.shape[0] == 0 or matrix.shape[1] == 0:
            return False, "Matrix cannot be empty!", None
            
        return True, "Data is valid.", matrix

    except Exception as e:
        return False, f"Data format error: {str(e)}", None