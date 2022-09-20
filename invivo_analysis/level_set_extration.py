import os
from os.path import join
import matplotlib.pyplot as plt
import seaborn as sns
# sns.set()
import pickle as pkl
import numpy as np
import pandas as pd
from easydict import EasyDict as edict
from scipy.stats import pearsonr, spearmanr
from skimage.measure import find_contours
from core.utils.plot_utils import saveallforms
from invivo_analysis.neural_data_lib import extract_meta_data
from invivo_analysis.neural_data_lib import get_Evol_Manif_stats, load_score_mat, mat_path
from invivo_analysis.level_set_lib import level_set_profile, plot_levelsets,\
    analyze_levelsets_topology, visualize_levelsets_all, plot_levelsets_topology
from invivo_analysis.Manif_interp_lib import sphere_interp_Manifold, \
    compute_all_meta, compute_all_interpolation, load_meta, load_data_interp

savedir = r"E:\OneDrive - Harvard University\Manifold_sphere_interp"
#%%

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
# Plot and Export topology related figures for individual experiments
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
#%% Extract the topological indices for each map.
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
syn_df.to_csv(join(savedir, "summary", "Both_topology_stats_synopsis.csv"))

#%% Evaluate the topological indices as a function of cortical level
valmsk = ~((syn_df.Animal == "Alfa") & (syn_df.Expi == 10))
df_summary = syn_df[valmsk].groupby("area", sort=False).agg(["mean", "sem"])
df_summary.T

#%% UMAP reduction of the topological signatures
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
umapdir = join(savedir, "summary", "topology_UMAP")
os.makedirs(umapdir, exist_ok=True)
#%%
umapper = UMAP(n_neighbors=10, min_dist=0.1, metric="manhattan", random_state=42)
umap_data = umapper.fit_transform(topofeatmat[valmsk])
#%%
figh, ax = plt.subplots(1, 1, figsize=(7, 7))
umap.plot.points(umapper, labels=syn_df.area[valmsk],
                 theme='viridis', width=600, height=600, alpha=0.8,
                 color_key={"V1": "#7920FF", "V4": "#2CA02C", "IT":"#FF7F0E"}, ax=ax)
saveallforms(umapdir, "topol_umap_area", figh)
figh.show()
#%%
for valuenm in ["branch_lowest_lvl", "loop_lowest_lvl", "branch_lowest_idx", "loop_lowest_idx",]:
    figh, ax = plt.subplots(1, 1, figsize=(7, 7))
    ax = umap.plot.points(umapper, values=syn_df[valuenm][valmsk], cmap="viridis",
                     width=600, height=600, alpha=0.8, ax=ax, background="black")
    # plt.colorbar(ax.collections[0], ax=ax)
    ax.set_title(f"Topological Signature UMAP\nValue colormap {valuenm}")
    saveallforms(umapdir, f"topol_umap_{valuenm}", figh)
    plt.show()
#%% export color map
for valuenm in ["branch_lowest_lvl", "loop_lowest_lvl", "branch_lowest_idx", "loop_lowest_idx",]:
    figh, ax = plt.subplots(1, 1, figsize=(7, 7))
    ax = umap.plot.points(umapper, values=syn_df[valuenm][valmsk], cmap="viridis",
                     width=600, height=600, alpha=0.8, ax=ax, background="black")
    plt.colorbar(ax.collections[0], ax=ax)
    ax.set_title(f"Topological Signature UMAP\nValue colormap {valuenm}")
    plt.tight_layout()
    saveallforms(umapdir, f"topol_umap_{valuenm}_colorbar", figh)
    plt.show()

#%%
plt.figure()
plt.scatter(umap_data[:, 0], umap_data[:, 1], c=syn_df.loop_lowest_lvl[valmsk], s=syn_df.area[valmsk], cmap="tab10")
plt.show()
#%%
#%%
print(spearmanr(syn_df.loop_lowest_idx[valmsk], umap_data[:, 0]))
print(spearmanr(syn_df.loop_lowest_idx[valmsk], umap_data[:, 1]))
#%%
spearmanr(syn_df.loop_lowest_idx[valmsk], syn_df.area_idx[valmsk])

