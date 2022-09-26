"""The library for computing level set, analysis
and computing the topological profile of the level set.
"""
from os.path import join
import matplotlib.pyplot as plt
import seaborn as sns
# sns.set()
import pickle as pkl
import numpy as np
import pandas as pd
from easydict import EasyDict as edict
from skimage.measure import find_contours
from scipy.interpolate import SmoothSphereBivariateSpline, RectSphereBivariateSpline
from invivo_analysis.neural_data_lib import load_score_mat, get_Evol_Manif_stats, mat_path

def is_close_loop(curve):
    """Check if the curve is a close loop.
    Curve is a Nxd array, each row is a point on the sphere.
    """
    return np.allclose(curve[0, :], curve[-1, :])


def level_set_profile(level_sets):
    """Compute the topological profile of level set at one level."""
    n_branch = len(level_sets)
    n_loop = 0
    n_line = 0
    for i in range(n_branch):
        if is_close_loop(level_sets[i]):
            n_loop += 1
        else:
            n_line += 1
    return n_branch, n_loop, n_line


def plot_levelsets(level_sets, actmap=None, ax=None, cmap=None, **kwargs):
    """Plot a list of Nx2 curves on the plane."""
    if ax is None:
        ax = plt.gca()
    if actmap is not None:
        im = ax.imshow(actmap, cmap=cmap)
        plt.colorbar(im, ax=ax)
    for curve in level_sets:
        ax.plot(curve[:, 1], curve[:, 0], "red", **kwargs)
    return ax


def curve_length(curve):
    """curve length in Euclidean L2 sense"""
    return np.sum(np.sqrt(np.sum(np.diff(curve, axis=0)**2, axis=1)))


def spherical_length(curve):
    pass


def analyze_levelsets_topology(data_interp, levels=None, nlevels=21, ):
    rngmax = data_interp.max()
    rngmin = data_interp.min()
    if levels is None:
        levels = np.linspace(data_interp.min(), data_interp.max(), nlevels)
    df_col = []
    lvlset_dict = {}
    for lvl in levels:
        lvlsets = find_contours(data_interp, lvl)
        nbr, nloop, nline = level_set_profile(lvlsets)
        df_col.append(edict(level=lvl, n_branch=nbr, n_loop=nloop, n_line=nline,
                            level_maxfrac=lvl / rngmax, ))
        lvlset_dict[lvl] = lvlsets
    df = pd.DataFrame(df_col)
    return df, lvlset_dict


def visualize_levelsets_all(data_interp, levels=None, nlevels=21, print_info=True, ):
    rngmax = data_interp.max()
    rngmin = data_interp.min()
    if levels is None:
        levels = np.linspace(data_interp.min(), data_interp.max(), nlevels)

    figh, axh = plt.subplots(1, 1, figsize=(8, 7))
    plt.imshow(data_interp, cmap="inferno")  # origin="lower")
    plt.xticks(range(0, 181, 45), range(-90, 91, 45))
    plt.yticks(range(0, 181, 45), range(-90, 91, 45))
    plt.colorbar()
    for lvl in levels:
        lvlsets = find_contours(data_interp, lvl)
        nbr, nloop, nline = level_set_profile(lvlsets)
        plot_levelsets(lvlsets, ax=axh)
        if print_info:
            print("level %.1f:\t%d branches, %d loops, %d lines" % (lvl, nbr, nloop, nline))

    return figh, axh


def plot_levelsets_topology(df, bslmean=None, explabel="", axh=None):
    """Plot the topology of level sets as a function of activation level"""
    from matplotlib.ticker import MaxNLocator
    if axh is None:
        figh, axh = plt.subplots(1, 1, figsize=(5, 4))
    else:
        figh = axh.figure
    axh.plot(df.level, df.n_branch, "o-", label="n_branch", alpha=0.5, lw=2)
    axh.plot(df.level, df.n_loop, "o-", label="n_loop", alpha=0.5, lw=2)
    axh.plot(df.level, df.n_line, "o-", label="n_line", alpha=0.5, lw=2)
    if bslmean is not None:
        axh.axvline(bslmean, color="k", linestyle="--", label="baseline")
    axh.set_xlabel("activation level (spk/s)")
    axh.set_ylabel("number of levelsets")
    axh.set_title("%s\nLevelset topology"%explabel)
    axh.legend()
    axh.yaxis.set_major_locator(MaxNLocator(integer=True))
    return figh, axh


import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
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


def plot_spherical_levelset(lvlset_dict):
    """Plot the level set on the sphere"""
    #TODO: color the level set by the value of the activation map
    cmap = plt.get_cmap('viridis')
    # normalize to the range of the activation map
    levels = [*lvlset_dict.keys()]
    lvlmax = np.max(levels)
    lvlmin = np.min(levels)
    fig = plt.figure(figsize=(8, 7))
    ax = fig.add_subplot(111, projection='3d')
    for lvl in levels:
        lvlcolor = cmap((lvl - lvlmin) / (lvlmax - lvlmin))
        lvlset = lvlset_dict[lvl]
        for idxsegm in lvlset:
            pnt = _idx2deg2pnt_vec(idxsegm)
            ax.scatter(pnt[:, 0], pnt[:, 1], pnt[:, 2],
                       s=9, color=lvlcolor, alpha=0.3)
            ax.plot(pnt[:, 0], pnt[:, 1], pnt[:, 2], color=lvlcolor,
                    alpha=0.3)
    ax.scatter(0, 0, 0, c="k", s=100)
    # ax.scatter(maxpnt[:, 0], maxpnt[:, 1], maxpnt[:, 2], c="r", s=200, marker="*")
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.set_zlim(-1, 1)
    ax.view_init(azim=-30, elev=10)  # set view
    ax.set_box_aspect((1, 1, 1))  # set aspect ratio
    plt.tight_layout()
    plt.show()
    return fig, ax

if __name__ == "__main__":
    #%%
    Animal = "Alfa"
    EStats, MStats = get_Evol_Manif_stats(Animal)
    #%%
    scorecol_M, imgfullpath_vect_M = load_score_mat(EStats, MStats, 20, "Manif_avg", wdws=[(50, 200)], stimdrive="S")
    scorecol_M_sgtr, _ = load_score_mat(EStats, MStats, 20, "Manif_sgtr", wdws=[(50, 200)], stimdrive="S")
    bslcol_M_sgtr, _ = load_score_mat(EStats, MStats, 20, "Manif_sgtr", wdws=[(0, 40)], stimdrive="S")
    #%%
    actmap = scorecol_M.reshape(11, 11)
    imgfp_mat = np.array(imgfullpath_vect_M).reshape((11, 11))
    bslvec = np.concatenate(bslcol_M_sgtr)
    bslmean = bslvec.mean()
    bslstd = bslvec.std()
    # 1st index is PC2 degree, 2nd index is PC3 degree
    # 1st index is Theta, 2nd index is Phi
    actmap_pole = actmap.copy()
    actmap_poleval0 = actmap[:, 0].mean()
    actmap_polevalpi = actmap[:, -1].mean()
    actmap_pole[:, 0] = actmap[:, 0].mean()
    actmap_pole[:, -1] = actmap[:, -1].mean()
    #%% Estimate variability from the single trial response.
    # mean std and sem.
    stdmean = np.mean([np.std(trrsp) for trrsp in scorecol_M_sgtr])
    semmean = np.mean([np.std(trrsp)/np.sqrt(len(trrsp)) for trrsp in scorecol_M_sgtr])
    # square of st error of the mean
    semsqarray = np.array([np.var(trrsp)/len(trrsp) for trrsp in scorecol_M_sgtr]).reshape(11,11)
    # sum of square of sem across the interpolated points
    semsqsum = semsqarray[:, 1:-1].sum()  # np.sum([np.var(trrsp) for trrsp in scorecol_M_sgtr])

    #%%
    # lats_vec = np.arange(0, 180.1, 18) / 180 * np.pi
    # lats_vec = np.linspace(0.1, 179.9, 11) / 180 * np.pi
    lats_vec = np.linspace(0, 180, 11)[1:-1] / 180 * np.pi
    lons_vec = np.arange(-90, 90.1, 18) / 180 * np.pi
    lut = RectSphereBivariateSpline(lats_vec, lons_vec, actmap[:, 1:-1].T,
            pole_values=(actmap_poleval0, actmap_polevalpi),
            pole_exact=False, s=semsqsum)

    #%%
    new_lats_vec = np.arange(0, 180.1, 1) / 180 * np.pi
    new_lons_vec = np.arange(-90, 90.1, 1) / 180 * np.pi
    new_lats, new_lons = np.meshgrid(new_lats_vec, new_lons_vec)
    data_interp = lut.ev(new_lats.ravel(),
                         new_lons.ravel()).reshape(new_lats.shape)
    #%%
    plt.figure(figsize=(12, 5))
    plt.subplot(121)
    plt.imshow(data_interp, )  # origin="lower")
    plt.colorbar()
    plt.subplot(122)
    plt.imshow(actmap, )  # origin="lower")
    plt.colorbar()
    plt.tight_layout()
    plt.show()