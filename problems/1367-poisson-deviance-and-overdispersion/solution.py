import numpy as np


def poisson_deviance(y: np.ndarray, mu: np.ndarray) -> float:
    """Poisson deviance, using the convention 0 * log(0) = 0."""
    # Your code here
    y_ = np.where(y!=0, y, 1)
    return 2 * (y * np.log(y_/mu) - (y-mu)).sum()


def dispersion_ratio(y: np.ndarray, mu: np.ndarray, n_params: int) -> float:
    """Pearson chi-square divided by (n - n_params)."""
    # Your code here
    return ((y-mu)**2 / mu).sum() / (len(y) - n_params)
