"""K-means++ initialization without recomputing distances to previous centres."""
import numpy as np
from scipy.cluster.vq import kmeans2
from scipy.spatial.distance import cdist


def inducing_kmeans(x, count, seed, iterations=20):
    """Same k-means++/Lloyd procedure and legacy seed semantics as SciPy 1.15.

    scipy.cluster.vq._kpp recomputes all previous centre distances on every
    seeding iteration. Maintaining their elementwise minimum instead changes
    seeding work from O(N*M^2*D) to O(N*M*D), without changing the choices.
    Lloyd iterations remain SciPy kmeans2 with the supplied initial matrix.
    """
    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 2 or not 1 <= count <= len(x):
        raise ValueError("Require a feature matrix and 1 <= inducing count <= N")
    if not np.isfinite(x).all():
        raise ValueError("Non-finite training features")
    rng = np.random.RandomState(seed)
    centres = np.empty((count, x.shape[1]), dtype=x.dtype)
    nearest = np.full(len(x), np.inf)
    for i in range(count):
        if i == 0:
            index = rng.randint(len(x))
        else:
            total = nearest.sum()
            if total <= 0:
                raise ValueError("Fewer distinct feature rows than inducing points")
            index = np.searchsorted((nearest/total).cumsum(), rng.uniform())
        centres[i] = x[index]
        if i < count-1:
            nearest = np.minimum(nearest, cdist(centres[i:i+1], x, metric="sqeuclidean")[0])
    return kmeans2(x, centres, iter=iterations, minit="matrix", seed=seed)[0]
