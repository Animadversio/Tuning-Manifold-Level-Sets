
from os.path import join
import matplotlib.pyplot as plt
import seaborn as sns
# sns.set()
import pickle as pkl
import numpy as np
import pandas as pd
from easydict import EasyDict as edict
from skimage.measure import find_contours
from invivo_analysis.neural_data_lib import extract_meta_data
from invivo_analysis.neural_data_lib import get_Evol_Manif_stats, load_score_mat, mat_path
from invivo_analysis.level_set_lib import sphere_interp_Manifold, \
    level_set_profile, plot_levelsets

savedir = r"E:\OneDrive - Harvard University\Manifold_sphere_interp"
#%% Mass calculation
for Animal in ["Alfa", "Beto"]:
    EStats, MStats = get_Evol_Manif_stats(Animal)
    for Expi in range(1, len(EStats)+1):
        data_interp, lut, actmap, bslmean = sphere_interp_Manifold(Animal, Expi, EStats, MStats)
        assert data_interp.min() >= 0
        np.savez(join(savedir, f"{Animal}_{Expi}_interp_map.npz"),
                 data_interp=data_interp, actmap=actmap, bslmean=bslmean)
        pkl.dump(lut, open(join(savedir, f"{Animal}_{Expi}_interp_lut.pkl"), "wb"))
#%%
meta_col = edict()
for Animal in ["Alfa", "Beto"]:
    EStats, MStats = get_Evol_Manif_stats(Animal)
    meta_col[Animal] = []
    for Expi in range(1, len(EStats)+1):
        meta, expstr = extract_meta_data(Animal, Expi, EStats, MStats)
        print(expstr)
        meta_col[Animal].append(meta)

pkl.dump(meta_col, open(join(savedir, "Both_metadata.pkl"), "wb"))
#%%

#%%
def load_data_interp(Animal, Expi, savedir=savedir):
    """ data loading routine from precomputed interpolated tuning map."""
    data = np.load(join(savedir, f"{Animal}_{Expi}_interp_map.npz"), allow_pickle=True)
    data_interp = data["data_interp"]
    actmap = data["actmap"]
    bslmean = data["bslmean"]
    lut = np.load(join(savedir, f"{Animal}_{Expi}_interp_lut.pkl"), allow_pickle=True)
    return data_interp, lut, actmap, bslmean


def load_meta(Animal, Expi, savedir=savedir):
    meta_col = pkl.load(open(join(savedir, "Both_metadata.pkl"), "rb"))
    return meta_col[Animal][Expi-1]
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
# Plot and Export topology related figures
sumdir = join(savedir, "summary", "topology")
lvldir = join(savedir, "levelsets")
for Animal in ["Alfa", "Beto"]:
    EStats, MStats = get_Evol_Manif_stats(Animal)
    for Expi in range(1, len(EStats)+1):
        explabel = "%s_Exp%02d" % (Animal, Expi)
        meta = load_meta(Animal, Expi, savedir=savedir)
        data_interp, lut, actmap, bslmean = load_data_interp(Animal, Expi, savedir=savedir)
        df, lvlset_dict = analyze_levelsets_topology(data_interp, nlevels=21)
        figh, axh = visualize_levelsets_all(data_interp, nlevels=21, print_info=False)
        saveallforms(sumdir, "%s_levelsets_all"%explabel, figh)
        figh2, axh2 = plot_levelsets_topology(df, bslmean, meta.expstr)
        saveallforms(sumdir, "%s_levelsets_topo_profile"%explabel, figh2)
        df.to_csv(join(lvldir, "%s_levelsets_topo_profile.csv"%explabel))
        pkl.dump(lvlset_dict, open(join(lvldir, "%s_levelsets.pkl"%explabel), "wb"))
        plt.close("all")
#%%
def get_lowest_single_level(df, tolerance=1):
    is_single_branch = df.n_branch == 1
    is_single_loop = df.n_loop == 1
    nlvl = df.shape[0]
    # for i in range(nlvl-1, 0, -1):
    branch_lowest_idx = nlvl - 1
    for i in reversed(df.index):
        if i == nlvl - 1: continue
        if is_single_branch[i]:
            continue
        else:
            branch_lowest_idx = i + 1
            break

    loop_lowest_idx = nlvl - 1
    for i in reversed(df.index):
        if i == nlvl - 1: continue
        if is_single_loop[i]:
            continue
        else:
            loop_lowest_idx = i + 1
            break

    return branch_lowest_idx, loop_lowest_idx

get_lowest_single_level(df, )
#%%
ExpNum = {"Alfa": 46, "Beto": 45}
syn_col = []
for Animal in ["Alfa", "Beto"]:
    for Expi in range(1, ExpNum[Animal]):
        meta = load_meta(Animal, Expi, savedir=savedir)
        data_interp, lut, actmap, bslmean = load_data_interp(Animal, Expi, savedir=savedir)
        df, lvlset_dict = analyze_levelsets_topology(data_interp, nlevels=21)
        nbranch_AUC = np.trapz(df.n_branch, df.level_maxfrac)
        nloop_AUC = np.trapz(df.n_loop, df.level_maxfrac)
        nline_AUC = np.trapz(df.n_line, df.level_maxfrac)
        nbranch_wsum = np.dot(df.n_branch, df.level_maxfrac)
        nloop_wsum = np.dot(df.n_loop, df.level_maxfrac)
        nline_wsum = np.dot(df.n_line, df.level_maxfrac)
        branch_lowest_idx, loop_lowest_idx = get_lowest_single_level(df, )
        branch_lowest_lvl, loop_lowest_lvl = df.level_maxfrac[branch_lowest_idx], df.level_maxfrac[loop_lowest_idx]
        S = edict(Animal=Animal, Expi=Expi, expstr=meta.expstr, area=meta.area,
                  nbranch_AUC=nbranch_AUC, nloop_AUC=nloop_AUC, nline_AUC=nline_AUC,
                  nbranch_wsum=nbranch_wsum, nloop_wsum=nloop_wsum, nline_wsum=nline_wsum,
                  branch_lowest_lvl=branch_lowest_lvl, loop_lowest_lvl=loop_lowest_lvl,
                  branch_lowest_idx=branch_lowest_idx, loop_lowest_idx=loop_lowest_idx)
        syn_col.append(S)
syn_df = pd.DataFrame(syn_col)
syn_df["area_idx"] = syn_df.area.map({"V1": 0, "V4": 1, "IT": 2})
#%%
valmsk = ~((syn_df.Animal == "Alfa") & (syn_df.Expi == 10))
df_summary = syn_df[valmsk].groupby("area", sort=False).agg(["mean", "sem"])
#%%
df_summary.T
#%%

#%% Collect feature set of topological signatures
topofeatmat = []
for Animal in ["Alfa", "Beto"]:
    for Expi in range(1, ExpNum[Animal]):
        meta = load_meta(Animal, Expi, savedir=savedir)
        data_interp, lut, actmap, bslmean = load_data_interp(Animal, Expi, savedir=savedir)
        df, lvlset_dict = analyze_levelsets_topology(data_interp, nlevels=21)
        featvec = np.concatenate([df.n_loop, df.n_line, ]) # df.n_branch,
        topofeatmat.append(featvec)
topofeatmat = np.array(topofeatmat)

#%%
from umap import UMAP
import umap.plot
umapper = UMAP(n_neighbors=10, min_dist=0.1, metric="manhattan", random_state=42)
umap_data = umapper.fit_transform(topofeatmat[valmsk])
#%%
figh, ax = plt.subplots(1, 1, figsize=(7, 7))
umap.plot.points(umapper, labels=syn_df.area[valmsk],
                 theme='viridis', width=600, height=600, alpha=0.8,
                 color_key={"V1": "#7920FF", "V4": "#2CA02C", "IT":"#FF7F0E"})
# saveallforms(umapdir, "topol_umap_area", figh)
plt.show()
#%%
figh, ax = plt.subplots(1, 1, figsize=(7, 7))
ax = umap.plot.points(umapper, values=syn_df.loop_lowest_lvl[valmsk], cmap="viridis",
                 width=600, height=600, alpha=0.8, ax=ax, background="black")
plt.colorbar(ax.collections[0], ax=ax)
# saveallforms(umapdir, "topol_umap_loop_lowest_lvl", figh)
plt.show()
#%%
plt.figure()
plt.scatter(umap_data[:, 0], umap_data[:, 1], c=syn_df.loop_lowest_lvl[valmsk], s=syn_df.area[valmsk], cmap="tab10")
plt.show()


