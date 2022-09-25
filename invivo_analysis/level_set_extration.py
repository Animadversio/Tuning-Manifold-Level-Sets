"""Script for extracting level sets from all Manifold experiments in vivo"""
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
