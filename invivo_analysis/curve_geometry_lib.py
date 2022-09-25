import numpy as np


def sphere_arc_dist2pnt(curve, pnt):
    """
    Compute the arc distance between a curve and a point on the unit sphere
    :param curve: np.array, (N, 3)
    :param pnt: np.array, (3,)
    :return: np.array, (N,)
    """
    return np.arccos(np.sum(curve * pnt, axis=1))


def sphere_arc_dist2curve(curve1, curve2):
    """
    Compute the arc distance between two curves on the unit sphere
    :param curve1: np.array, (N, 3)
    :param curve2: np.array, (M, 3)
    :return: np.array, (N, M)
    """
    return np.arccos(np.dot(curve1, curve2.T))


def _idx2deg2pnt_vec(idx_curve):
    """ Convert degree in [0,180]x[0,180] to a point on the sphere
    :param idx_curve: np.array, (N, 2).
                Will format as such if it's 1d tuple, or list
    :return: np.array, (N, 3)
    """
    # rad = idx_curve
    if type(idx_curve) is not np.ndarray:
        idx_curve = np.array(idx_curve)
    if idx_curve.ndim == 1:
        idx_curve = idx_curve[np.newaxis, :]
    deg = idx_curve - 90
    rad = deg / 180 * np.pi
    return np.array([np.cos(rad[:, 1]) * np.cos(rad[:, 0]),
                     np.cos(rad[:, 1]) * np.sin(rad[:, 0]),
                     np.sin(rad[:, 1])]).T