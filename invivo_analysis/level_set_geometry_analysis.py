import numpy as np
import pandas as pd
from easydict import EasyDict as edict
from scipy.stats import pearsonr, spearmanr
from skimage.measure import find_contours
from core.utils.plot_utils import saveallforms
from invivo_analysis.neural_data_lib import extract_meta_data
from invivo_analysis.neural_data_lib import get_Evol_Manif_stats, load_score_mat, mat_path, ExpNum
from invivo_analysis.level_set_lib import level_set_profile, plot_levelsets,\
    analyze_levelsets_topology, visualize_levelsets_all, plot_levelsets_topology, \
    is_close_loop
from invivo_analysis.Manif_interp_lib import sphere_interp_Manifold, \
    compute_all_meta, compute_all_interpolation, load_meta, load_data_interp
savedir = r"E:\OneDrive - Harvard University\Manifold_sphere_interp"
#%%
for Animal in ["Alfa", "Beto"]:
    for Expi in range(1, ExpNum[Animal]+1):
        meta = load_meta(Animal, Expi, savedir=savedir)
        data_interp, lut, actmap, bslmean = load_data_interp(Animal, Expi, savedir=savedir)
        df, lvlset_dict = analyze_levelsets_topology(data_interp, nlevels=21)
#%%
figh, ax = visualize_levelsets_all(data_interp, nlevels=21, print_info=False)
figh.show()
#%% 3d spherical geometry routines
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
#%%
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
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

#%%
maxidx = np.unravel_index(data_interp.argmax(), data_interp.shape)
maxpnt_sph = _idx2deg2pnt_vec(maxidx)

levels = [*lvlset_dict.keys()]
plt.figure()
for lvl in levels:
    lvlset = lvlset_dict[lvl]
    for segm in lvlset:
        isloop = is_close_loop(segm)
        lvlset_sph = _idx2deg2pnt_vec(segm)
        dist = sphere_arc_dist2pnt(lvlset_sph, maxpnt_sph)
        # plt.plot(dist, linestyle="-" if isloop else "--")
        # if isloop:
        plt.scatter(np.ones(dist.shape) * lvl, dist, s=9, alpha=0.5)
        plt.scatter(lvl, max(dist)/min(dist), s=9, alpha=0.5)
plt.show()
#%%
lvlmax = np.max(levels)
maxidx = np.unravel_index(data_interp.argmax(), data_interp.shape)
maxpnt_sph = _idx2deg2pnt_vec(maxidx)
levels = [*lvlset_dict.keys()]
geom_col = []
for lvl in levels:
    lvlset = lvlset_dict[lvl]
    for segm in lvlset:
        isloop = is_close_loop(segm)
        isinside = isloop and (maxidx[0] < segm[:, 0].max()) and (maxidx[0] > segm[:, 0].min()) \
                          and (maxidx[1] < segm[:, 1].max()) and (maxidx[1] > segm[:, 1].min())
        lvlset_sph = _idx2deg2pnt_vec(segm)
        dist = sphere_arc_dist2pnt(lvlset_sph, maxpnt_sph)
        distmat = sphere_arc_dist2curve(lvlset_sph, lvlset_sph)
        geom_col.append(edict({"level": lvl, "isloop": isloop, "isinside": isloop and True,
                               "angdist_max": max(dist), "angdist_min": min(dist),
                               "angdist_mean": np.mean(dist), "angdist_std": np.std(dist),
                               "angdist_ratio": max(dist)/min(dist),
                               "level_maxnorm": lvl / lvlmax,
                               "distmat_max": distmat.max()}))
df_geom = pd.DataFrame(geom_col)
#%%
import seaborn as sns
sns.scatterplot(x="level", y="angdist_ratio", data=df_geom, hue="isloop")
# sns.pointplot(x="level", y="angdist_ratio", hue="isloop", data=df_geom, join=False)
plt.show()
#%%
figh, axh = plt.subplots(1, 1, figsize=(5, 4))
sns.scatterplot(x="level", y="angdist_min", data=df_geom, hue="isloop", ax=axh, alpha=0.6)
sns.scatterplot(x="level", y="angdist_max", data=df_geom, hue="isloop", ax=axh, alpha=0.6)
sns.scatterplot(x="level", y="angdist_mean", data=df_geom, hue="isloop", ax=axh, alpha=0.9)
plt.scatter(lvlmax, 0, c="k", s=64)
axh.axvline(bslmean, linestyle="--", color="k")
axh.set_xlabel("Level")
axh.set_ylabel("Angular distance (rad)")
axh.set_title(meta.expstr)
figh.show()
#%%
figh, axh = plt.subplots(1,1, figsize=(8, 6))
df_geom.plot("level", "angdist_mean", c=df_geom.isloop,
             kind="scatter", cmap='Accent', ax=axh) #
df_geom.plot("level", "angdist_min", c=df_geom.isloop,
             kind="scatter", cmap='Accent', ax=axh) #
df_geom.plot("level", "angdist_max", c=df_geom.isloop,
             kind="scatter", cmap='Accent', ax=axh) #
plt.show()
#%%
plot_spherical_levelset(lvlset_dict)

ax.scatter(maxpnt[:, 0], maxpnt[:, 1], maxpnt[:, 2], c="r", s=200, marker="*")

#%%


