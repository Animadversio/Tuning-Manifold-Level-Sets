
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
#%%
#%%
def sphere_interp_actmap(actmap, s, ):
    """Simple version of activation map interpolation."""
    actmap_poleval0 = actmap[:, 0].mean()
    actmap_polevalpi = actmap[:, -1].mean()
    lats_vec = np.linspace(0, 180, 11)[1:-1] / 180 * np.pi
    lons_vec = np.arange(-90, 90.1, 18) / 180 * np.pi
    lut = RectSphereBivariateSpline(lats_vec, lons_vec, actmap[:, 1:-1].T,
                                    pole_values=(actmap_poleval0, actmap_polevalpi),
                                    pole_exact=False, s=s)
    new_lats_vec = np.arange(0, 180.1, 1) / 180 * np.pi
    new_lons_vec = np.arange(-90, 90.1, 1) / 180 * np.pi
    new_lats, new_lons = np.meshgrid(new_lats_vec, new_lons_vec)
    data_interp = lut.ev(new_lats.ravel(),
                         new_lons.ravel()).reshape(new_lats.shape)
    return data_interp


def sphere_interp_Manifold(Animal, Expi, EStats=None, MStats=None, ):
    if EStats is None or MStats is None:
        EStats, MStats = get_Evol_Manif_stats(Animal)
    # Load the response data
    scorecol_M, imgfullpath_vect_M = load_score_mat(EStats, MStats, Expi, "Manif_avg", wdws=[(50, 200)], stimdrive="S")
    scorecol_M_sgtr, _ = load_score_mat(EStats, MStats, Expi, "Manif_sgtr", wdws=[(50, 200)], stimdrive="S")
    bslcol_M_sgtr, _ = load_score_mat(EStats, MStats, Expi, "Manif_sgtr", wdws=[(0, 40)], stimdrive="S")
    # 1st index is PC2 degree, 2nd index is PC3 degree
    # 1st index is Theta, 2nd index is Phi
    # format the data into map
    actmap = scorecol_M.reshape(11, 11)
    actcolmap = np.array(scorecol_M_sgtr, dtype=object).reshape(11, 11)
    imgfp_mat = np.array(imgfullpath_vect_M).reshape((11, 11))
    bslvec = np.concatenate(bslcol_M_sgtr)
    bslmean = bslvec.mean()
    bslstd = bslvec.std()
    #% Estimate variability from the single trial response.
    # square of st error of the mean
    semsqarray = np.array([np.var(trrsp) / len(trrsp) for trrsp in scorecol_M_sgtr]).reshape(11, 11)
    # sum of square of sem across the interpolated points
    semsqsum = semsqarray[:, 1:-1].sum()  # np.sum([np.var(trrsp) for trrsp in scorecol_M_sgtr])
    #%  Norse pole and south pole value
    actmap_poleval0 = np.concatenate(actcolmap[:, 0]).mean()  # actmap[:, 0].mean()
    actmap_polevalpi = np.concatenate(actcolmap[:, -1]).mean()  # actmap[:, -1].mean()
    #%
    lats_vec = np.linspace(0, 180, 11)[1:-1] / 180 * np.pi
    lons_vec = np.arange(-90, 90.1, 18) / 180 * np.pi
    lut = RectSphereBivariateSpline(lats_vec, lons_vec, actmap[:, 1:-1].T,
                                    pole_values=(actmap_poleval0, actmap_polevalpi),
                                    pole_exact=False, s=semsqsum)
    new_lats_vec = np.arange(0, 180.1, 1) / 180 * np.pi
    new_lons_vec = np.arange(-90, 90.1, 1) / 180 * np.pi
    new_lats, new_lons = np.meshgrid(new_lats_vec, new_lons_vec)
    data_interp = lut.ev(new_lats.ravel(),
                         new_lons.ravel()).reshape(new_lats.shape)
    data_interp = np.maximum(data_interp, 0.0)
    return data_interp, lut, actmap, bslmean


def is_close_loop(curve):
    return np.allclose(curve[0, :], curve[-1, :])


def level_set_profile(level_sets):
    n_branch = len(level_sets)
    n_loop = 0
    n_line = 0
    for i in range(n_branch):
        if is_close_loop(level_sets[i]):
            n_loop += 1
        else:
            n_line += 1
    return n_branch, n_loop, n_line


def plot_levelsets(level_sets, actmap=None, ax=None, **kwargs):
    if ax is None:
        ax = plt.gca()
    if actmap is not None:
        im = ax.imshow(actmap, **kwargs)
        plt.colorbar(im, ax=ax)
    for curve in level_sets:
        ax.plot(curve[:, 1], curve[:, 0], "red")
    return ax


def curve_length(curve):
    return np.sum(np.sqrt(np.sum(np.diff(curve, axis=0)**2, axis=1)))


def spherical_length(curve):
    pass




#%% Mass calculation
savedir = r"E:\OneDrive - Harvard University\Manifold_sphere_interp"
for Animal in ["Alfa", "Beto"]:
    EStats, MStats = get_Evol_Manif_stats(Animal)
    for Expi in range(1, len(EStats)+1):
        data_interp, lut, actmap, bslmean = sphere_interp_Manifold(Animal, Expi, EStats, MStats)
        assert data_interp.min() >= 0
        np.savez(join(savedir, f"{Animal}_{Expi}_interp_map.npz"),
                 data_interp=data_interp, actmap=actmap, bslmean=bslmean)
        pkl.dump(lut, open(join(savedir, f"{Animal}_{Expi}_interp_lut.pkl"), "wb"))
#%%
Animal = "Alfa"
EStats, MStats = get_Evol_Manif_stats(Animal)
data_interp, lut, actmap, bslmean = sphere_interp_Manifold(Animal, 20, EStats, MStats)
#%%
rngmax = data_interp.max()
rngmin = data_interp.min()
#%%
lvlsets = find_contours(actmap_pole, rngmax-5)
print("%d branches, %d loops, %d lines" % level_set_profile(lvlsets))
plot_levelsets(lvlsets, actmap_pole)
plt.show()
#%%
lvlsets = find_contours(data_interp, rngmax-4.5)
print("%d branches, %d loops, %d lines" % level_set_profile(lvlsets))
plot_levelsets(lvlsets, data_interp)
plt.show()
#%%
levels = np.linspace(rngmin, rngmax, 21)
figh, axh = plt.subplots(1, 1, figsize=(8, 7))
plt.imshow(data_interp, cmap="inferno")  # origin="lower")
plt.colorbar()
for lvl in levels:
    lvlsets = find_contours(data_interp, lvl)
    nbr, nloop, nline = level_set_profile(lvlsets)
    plot_levelsets(lvlsets, ax=axh)
    print("level %.1f:\t%d branches, %d loops, %d lines" % (lvl, nbr, nloop, nline))

plt.show()
#%%
def load_data_interp(Animal, Expi, ):
    data = np.load(join(savedir, f"{Animal}_{Expi}_interp_map.npz"), allow_pickle=True)
    data_interp = data["data_interp"]
    actmap = data["actmap"]
    bslmean = data["bslmean"]
    lut = np.load(join(savedir, f"{Animal}_{Expi}_interp_lut.pkl"), allow_pickle=True)
    return data_interp, lut, actmap, bslmean

#%%
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
    plt.colorbar()
    for lvl in levels:
        lvlsets = find_contours(data_interp, lvl)
        nbr, nloop, nline = level_set_profile(lvlsets)
        plot_levelsets(lvlsets, ax=axh)
        if print_info:
            print("level %.1f:\t%d branches, %d loops, %d lines" % (lvl, nbr, nloop, nline))

    return figh, axh


def plot_levelsets_topology(df, bslmean=None, explabel="", axh=None):
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

#%%
Animal, Expi = "Alfa", 3
explabel = "%s_Exp%02d" % (Animal, Expi)
data_interp, lut, actmap, bslmean = load_data_interp(Animal, Expi)
df, lvlset_dict = analyze_levelsets_topology(data_interp, nlevels=21)
figh, axh = visualize_levelsets_all(data_interp, nlevels=21, print_info=False)
figh.show()
figh2, axh2 = plot_levelsets_topology(df, bslmean, explabel)
figh2.show()
#%%
from core.utils.plot_utils import saveallforms
# Export topology related figures 
sumdir = join(savedir, "summary", "topology")
for Animal in ["Alfa", "Beto"]:
    EStats, MStats = get_Evol_Manif_stats(Animal)
    for Expi in range(1, len(EStats)+1):
        explabel = "%s_Exp%02d" % (Animal, Expi)
        data_interp, lut, actmap, bslmean = load_data_interp(Animal, Expi)
        df, lvlset_dict = analyze_levelsets_topology(data_interp, nlevels=21)
        figh, axh = visualize_levelsets_all(data_interp, nlevels=21, print_info=False)
        saveallforms(sumdir, "%s_levelsets_all"%explabel, figh)
        figh2, axh2 = plot_levelsets_topology(df, bslmean, explabel)
        saveallforms(sumdir, "%s_levelsets_topo_profile"%explabel, figh2)
#%%
lvlsets = find_contours(data_interp, rngmax - 1)
plt.figure(figsize=(6, 5))
plt.subplot(111)
plt.imshow(data_interp, )  # origin="lower")
plt.plot(lvlsets[0][:, 1], lvlsets[0][:, 0], "red")
plt.colorbar()
plt.show()

#%%
lvlsets = find_contours(actmap_pole, rngmax-5)
plt.figure(figsize=(6, 5))
plt.subplot(111)
plt.imshow(actmap_pole, )  # origin="lower")
plt.plot(lvlsets[0][:, 1], lvlsets[0][:, 0], "red")
plt.colorbar()
plt.show()
