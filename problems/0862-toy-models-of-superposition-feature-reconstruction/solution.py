import numpy as np

def superposition_reconstruct(W, b, X):
    """
    Compute reconstructed features for the toy superposition model.

    Args:
        W: array of shape (n_hidden, n_features)
        b: array of shape (n_features,)
        X: array of shape (batch_size, n_features)

    Returns:
        list of lists of shape (batch_size, n_features) with reconstructed features
    """
    W_np = np.asarray(W)
    b_np = np.asarray(b)
    X_np = np.asarray(X)

    return np.maximum(0, W_np.T @ W_np @ X_np.T + b_np[:, None]).T.tolist()
