import numpy as np
from scipy.stats import entropy


def symmetric_kl(p, q, bins=100, shared_bins=True):
    """
    Compute symmetric KL divergence between two distributions (histogram-based).
    """
    if not shared_bins:
        p_hist, bin_edges = np.histogram(p, bins=bins, density=True)
    else:
        # determine common range
        min_val = min(np.min(p), np.min(q))
        max_val = max(np.max(p), np.max(q))

        # create shared bin edges
        bin_edges = np.linspace(min_val, max_val, bins + 1)

        # compute histogram of p using pre-computed bins
        p_hist, _ = np.histogram(p, bins=bin_edges, density=True)

    # either way, compute histogram of q using the same bins as p
    q_hist, _ = np.histogram(q, bins=bin_edges, density=True)

    # avoid division-by-zero inside KL: add small constant and renormalise
    eps = 1e-12
    p_hist = np.clip(p_hist, eps, None)
    q_hist = np.clip(q_hist, eps, None)

    p_hist /= p_hist.sum()
    q_hist /= q_hist.sum()

    # compute KL divergences in both directions
    kl_pq = entropy(p_hist, q_hist)
    kl_qp = entropy(q_hist, p_hist)

    # return the average
    return 0.5 * (kl_pq + kl_qp)


def bootstrap_symmetric_kl(a, b, bins=100, n_bootstrap=1000, random_state=None):
    """
    Estimate variance of symmetric KL divergence using bootstrapping.

    Parameters
    ----------
    a, b : array-like
        Input samples (likelihoods).
    bins : int
        Number of bins for histogram discretisation.
    n_bootstrap : int
        Number of bootstrap resamples.
    random_state : int, optional
        Random seed for reproducibility.

    Returns
    -------
    mean : float
        Mean symmetric KL divergence across bootstraps.
    std : float
        Standard deviation of the bootstrap estimates.
    ci : tuple
        2.5 and 97.5 percentile confidence interval.
    samples : np.ndarray
        All bootstrap SKL values.
    """
    rng = np.random.default_rng(random_state)
    skl_samples = np.zeros(n_bootstrap)

    n_a, n_b = len(a), len(b)
    for i in range(n_bootstrap):
        a_resample = rng.choice(a, size=n_a, replace=True)
        b_resample = rng.choice(b, size=n_b, replace=True)
        skl_samples[i] = symmetric_kl(a_resample, b_resample, bins=bins)

    mean = skl_samples.mean()
    std = skl_samples.std()
    ci = np.percentile(skl_samples, [2.5, 97.5])

    return mean, std, ci, skl_samples
