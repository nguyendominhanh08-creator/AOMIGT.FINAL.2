def find_saddle_points(A):
    m = len(A)
    n = len(A[0])

    # Step 1: Find the minimum value of each row
    row_min = [min(row) for row in A]

    # Step 2: Find the maximum value of each column
    col_max = [
        max(A[i][j] for i in range(m))
        for j in range(n)
    ]

    # Step 3: Calculate Maximin and Minimax
    maximin = max(row_min)
    minimax = min(col_max)

    print("Payoff Matrix:")
    for row in A:
        print(row)

    print("Row minimums:", row_min)
    print("Column maximums:", col_max)
    print("Maximin =", maximin)
    print("Minimax =", minimax)

    # Step 4: Find all saddle points
    saddle_points = []

    for i in range(m):
        for j in range(n):
            if A[i][j] == row_min[i] and A[i][j] == col_max[j]:
                saddle_points.append((i + 1, j + 1, A[i][j]))

    # Step 5: Display results
    if maximin == minimax:
        print("Saddle point exists.")
        print("Game value =", maximin)
        print("Saddle point(s):")

        for row, col, value in saddle_points:
            print(f"Row {row}, Column {col}, Value = {value}")
    else:
        print("No pure-strategy saddle point exists.")

    return saddle_points


# Example
A = [
    [3, 1, 2],
    [4, 5, 6],
    [2, 0, 4]
]

find_saddle_points(A)
