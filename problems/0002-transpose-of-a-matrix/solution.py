def transpose_matrix(a: list[list[int|float]]) -> list[list[int|float]]:
    """
    Transpose a 2D matrix by swapping rows and columns.
    
    Args:
        a: A 2D matrix of shape (m, n)
    
    Returns:
        The transposed matrix of shape (n, m)
    """
    a_trans = []
    for i in range(len(a[0])):
        tmp = []
        for j in range(len(a)):
            tmp.append(a[j][i])
        a_trans.append(tmp)
    # Your code here
    return a_trans