"""An example of level set extraction and computation analysis on in vivo data."""

from os.path import join
import matplotlib.pyplot as plt
import seaborn as sns
# sns.set()
import pickle as pkl
import numpy as np
import pandas as pd
from easydict import EasyDict as edict
from skimage.measure import find_contours
from invivo_analysis.neural_data_lib import load_score_mat, get_Evol_Manif_stats, mat_path
from invivo_analysis.level_set_lib import sphere_interp_Manifold, \
    level_set_profile, plot_levelsets


Animal = "Alfa"
EStats, MStats = get_Evol_Manif_stats(Animal)
data_interp, lut, actmap, bslmean = sphere_interp_Manifold(Animal, 20, EStats, MStats)
#%%
rngmax = data_interp.max()
rngmin = data_interp.min()
#%% Contours from the original actmap
lvlsets = find_contours(actmap, rngmax-5)
print("%d branches, %d loops, %d lines" % level_set_profile(lvlsets))
plot_levelsets(lvlsets, actmap)
plt.show()
#%% Contours from the interpolated actmap
lvlsets = find_contours(data_interp, rngmax-4.5)
print("%d branches, %d loops, %d lines" % level_set_profile(lvlsets))
plot_levelsets(lvlsets, data_interp)
plt.show()
#%% Plot all level sets
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
#%% High light one level set
lvlsets = find_contours(data_interp, rngmax - 1)
plt.figure(figsize=(6, 5))
plt.subplot(111)
plt.imshow(data_interp, )  # origin="lower")
plt.plot(lvlsets[0][:, 1], lvlsets[0][:, 0], "red")
plt.colorbar()
plt.show()

#%%
lvlsets = find_contours(actmap, rngmax-5)
plt.figure(figsize=(6, 5))
plt.subplot(111)
plt.imshow(actmap, )  # origin="lower")
plt.plot(lvlsets[0][:, 1], lvlsets[0][:, 0], "red")
plt.colorbar()
plt.show()