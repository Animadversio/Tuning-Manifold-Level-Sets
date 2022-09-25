"""Visualize image curves in level set"""
import os
from os.path import join
import numpy as np
import pandas as pd
import seaborn as sns
from easydict import EasyDict as edict
from scipy.stats import pearsonr, spearmanr
from skimage.measure import find_contours
from core.utils.GAN_utils import upconvGAN
from core.utils.plot_utils import saveallforms, show_imgrid, save_imgrid
from invivo_analysis import *
from invivo_analysis.curve_geometry_lib import *
from invivo_analysis.curve_geometry_lib import _idx2deg2pnt_vec
from invivo_analysis.neural_data_lib import extract_meta_data, ExpNum, get_Basis_stats

#%%
def level_set_images(G, basis, spherenorm, levelset, Nsegs=21):
    pnts = _idx2deg2pnt_vec(levelset)
    segment_len = np.arccos(np.diag(pnts @ pnts.T, k=1))
    # arc distiance
    curv_cumlen = np.cumsum(segment_len)
    # L2 distance
    # curvlen_L2 = np.linalg.norm(pnts[1:] - pnts[:-1], axis=1).sum()
    curvlen = curv_cumlen[-1]
    node_len = np.linspace(0, curvlen, Nsegs)
    ninsert_index = np.searchsorted(curv_cumlen, node_len)
    sample_codes = pnts[ninsert_index] @ basis * spherenorm
    imgs = G.visualize_batch_np(sample_codes)
    return imgs
#%%
G = upconvGAN().cuda()
G.eval().requires_grad_(False)
#%%
savedir = r"E:\OneDrive - Harvard University\Manifold_sphere_interp"
rfmskdir  = r"E:\OneDrive - Harvard University\Manifold_attrb_mask"
imsavedir = r"E:\OneDrive - Harvard University\Manifold_sphere_interp\levelset_images"
os.makedirs(imsavedir, exist_ok=True)
for Animal in ["Alfa", "Beto"]:
    basisStats, ReprStats = get_Basis_stats(Animal)
    for Expi in range(1, ExpNum[Animal]+1):
        basis = basisStats[Expi - 1].basis_23
        spherenorm = basisStats[Expi - 1].sphere_norm
        meta = load_meta(Animal, Expi, savedir=savedir)
        data_interp, lut, actmap, bslmean = load_data_interp(Animal, Expi, savedir=savedir)
        df, lvlset_dict = analyze_levelsets_topology(data_interp, nlevels=21)
        rfdata = np.load(join(rfmskdir, f"{Animal}_Exp{Expi:02d}_mask_L2.npz"), allow_pickle=True)
        rfmask = rfdata['alphamsk_resize']
        for lvli, (level, lvlsets) in enumerate(lvlset_dict.items()):
            for segi, lvlset in enumerate(lvlsets):
                # print(lvlset)
                imgs = level_set_images(G, basis, spherenorm, lvlset, Nsegs=20)
                save_imgrid(imgs, join(imsavedir, f"{Animal}_Exp{Expi:02d}_contour_imgs_{lvli}-{segi}.jpg"), nrow=10, figsize=(10, 10))
                save_imgrid(imgs * rfmask, join(imsavedir, f"{Animal}_Exp{Expi:02d}_contour_imgs_{lvli}-{segi}_rfmsk.jpg"), nrow=10, figsize=(10, 10))


#%%
#%%

#%%
pnts = _idx2deg2pnt_vec(lvlset)
segment_len = np.arccos(np.diag(pnts @ pnts.T, k=1))
# arc distiance
curv_cumlen = np.cumsum(segment_len)
# L2 distance
# curvlen_L2 = np.linalg.norm(pnts[1:] - pnts[:-1], axis=1).sum()
curvlen = curv_cumlen[-1]
Nsegs = 21
node_len = np.linspace(0, curvlen, Nsegs)
ninsert_index = np.searchsorted(curv_cumlen, node_len)
# curvlen = np.arccos(np.diag(pnts @ pnts.T, k=1)).sum()
#%%
#%%
basis = basisStats[Expi - 1].basis_23
spherenorm = basisStats[Expi - 1].sphere_norm
sample_codes = pnts[ninsert_index] @ basis * spherenorm
imgs = G.visualize_batch_np(sample_codes)
#%%
show_imgrid(imgs, nrow=6, figsize=(10, 10))
#%%


