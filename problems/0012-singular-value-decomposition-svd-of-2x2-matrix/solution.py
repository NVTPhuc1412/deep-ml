import numpy as np

def svd_2x2_singular_values(A: np.ndarray) -> tuple:
    """
    Compute SVD of a 2x2 matrix using one Jacobi rotation.
    
    Args:
        A: A 2x2 numpy array
    
    Returns:
        Tuple (U, S, Vt) where A ≈ U @ diag(S) @ Vt
        - U: 2x2 orthogonal matrix
        - S: length-2 array of singular values
        - Vt: 2x2 orthogonal matrix (transpose of V)
    """
    # Your code here
    B = A.T @ A
    theta = np.arctan2(
        2 * B[0,1], 
        (B[0,0] - B[1,1])
        ) / 2
    
    sin = np.sin(theta)
    cos = np.cos(theta)
    V = np.array(
        [[cos, -sin],
        [sin, cos]]
    )
    D = V.T @ B @ V
    S = np.sqrt(np.maximum(np.diag(D), 0.0))
    
    order = np.argsort(S)[::-1]
    S, V = S[order], V[:, order]

    tol = 8 * np.finfo(float).eps * max(1.0, S[0])

    if S[0] <= tol:
        U = np.eye(2)
    elif S[1] <= tol:
        U = np.zeros(2,2)
        U[:, 0] = (A @ V[:, 0]) / S[0]
        U[:, 0] /= np.linalg.norm(U[:, 0])
        U[:, 1] = np.array([-U[1, 0], U[0, 0]])
    else:
        U = A @ V @ np.linalg.inv(np.diag(S))
    
    return (U, S, V.T)