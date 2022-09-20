import numpy as np
import pandas as pd
from easydict import EasyDict as edict
from scipy.stats import pearsonr, spearmanr
from skimage.measure import find_contours
from core.utils.plot_utils import saveallforms
from invivo_analysis.neural_data_lib import extract_meta_data
from invivo_analysis.neural_data_lib import get_Evol_Manif_stats, load_score_mat, mat_path, ExpNum
from invivo_analysis.level_set_lib import level_set_profile, plot_levelsets,\
    analyze_levelsets_topology, visualize_levelsets_all, plot_levelsets_topology
from invivo_analysis.Manif_interp_lib import sphere_interp_Manifold, \
    compute_all_meta, compute_all_interpolation, load_meta, load_data_interp
savedir = r"E:\OneDrive - Harvard University\Manifold_sphere_interp"

for Animal in ["Alfa", "Beto"]:
    for Expi in range(1, ExpNum[Animal]):
        meta = load_meta(Animal, Expi, savedir=savedir)
        data_interp, lut, actmap, bslmean = load_data_interp(Animal, Expi, savedir=savedir)
        df, lvlset_dict = analyze_levelsets_topology(data_interp, nlevels=21)