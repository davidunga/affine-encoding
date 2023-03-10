
import numpy as np
from scipy.ndimage import convolve1d


def mean_filter1d(x, w: int, axis=-1):
    return convolve1d(x, np.ones(w), axis=axis) / w


def zscore(xs: np.ndarray, x: np.ndarray | float = None, axis=-1, robust=False):
    """
    zscore(xs) - standardize xs.
    zscore(xs, x) - zscore of x relative to xs population. x has to be either a scalar, or same size as xs.
    zscore(__, robust=True) - compute robust zscore based on median absolute deviation
    """

    if robust:
        loc = np.median(xs, axis=axis, keepdims=True)
        scale = 1.4826 * np.median(np.abs(loc - xs), axis=axis, keepdims=True)
    else:
        loc = np.mean(xs, axis=axis, keepdims=True)
        scale = np.std(xs, axis=axis, keepdims=True)

    if x is None:
        return (xs - loc) / np.maximum(scale, np.finfo(float).eps)
    return (x - loc) / np.maximum(scale, np.finfo(float).eps)


def normalize(x: np.ndarray, axis=-1, kind="std"):
    """
    normalize array x along axis. normalization kinds:
        "std"       - zero mean, unit std
        "mad"       - zero median, unit median absolute deviation
        "max"       - min=0, max=1
        "sum"       - unit sum
        "sumabs"    - unit sum of absolutes
        "norm"      - unit l2 norm
    """

    match kind:
        case "std":
            return zscore(x, axis=axis, robust=False)
        case "mad":
            return zscore(x, axis=axis, robust=True)
        case "max":
            mn = np.min(x, axis=axis, keepdims=True)
            return (x - mn) / (np.max(x, axis=axis, keepdims=True) - mn)
        case "sum":
            return x / np.sum(x, axis=axis, keepdims=True)
        case "sumabs":
            return x / np.sum(np.abs(x), axis=axis, keepdims=True)
        case "norm":
            return x / np.linalg.norm(axis=axis, keepdims=True)
        case _:
            raise ValueError("Unknown normalization kind " + (kind if isinstance(kind, str) else ""))

