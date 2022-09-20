import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
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
#%%
Animal = "Alfa"
Expi = 3
EStats, MStats = get_Evol_Manif_stats("Alfa")
#%%
scorevec_ref, imgfp_ref = load_score_mat(EStats, MStats, 3, "EvolRef_avg", stimdrive="S")
scoresgtr_ref, _ = load_score_mat(EStats, MStats, 3, "EvolRef_sgtr", stimdrive="S")
#%
scorevec_manif, imgfp_manif = load_score_mat(EStats, MStats, 3, "Manif_avg", stimdrive="S")
scoresgtr_manif, _ = load_score_mat(EStats, MStats, 3, "Manif_sgtr", stimdrive="S")
#%%
sortidx = np.argsort(scorevec_ref)[::-1]
score_sorted = scorevec_ref[sortidx]
imgfp_ref_sorted = np.array(imgfp_ref)[sortidx]
#%%
mean_trlN = np.mean([len(A) for A in scoresgtr_manif])
scorevec_ref_init = np.array([np.mean(A[:round(mean_trlN)]) for A in scoresgtr_ref])
sortidx_init = np.argsort(scorevec_ref_init)[::-1]
score_init_sorted = scorevec_ref_init[sortidx_init]
#%%
import matplotlib.pyplot as plt
from skimage.transform import resize
from core.utils.montage_utils import build_montages, \
    color_framed_montages, make_grid_np
#%%
def load_img_seq(imgfp, idxs):
    img_col = []
    for i in range(len(idxs)):
        img = plt.imread(imgfp[idxs[i]])
        img_rs = resize(img, (256, 256), anti_aliasing=True)
        if img_rs.ndim == 2:
            img_rs = np.stack([img_rs]*3, axis=-1)
        if img_rs.shape[-1] == 4:
            img_rs = img_rs[:, :, :3]
        img_col.append(img_rs)
    return img_col
img_col = load_img_seq(imgfp_ref, sortidx[:6])
img_col_init = load_img_seq(imgfp_ref, sortidx_init[:6])

#%%
from core.utils.colormap_matlab import parula
vmin, vmax = scorevec_manif.min(), scorevec_manif.max()
mtg = build_montages(img_col, (256, 256), (3, 2), )
mtg_frm = color_framed_montages(img_col, (256, 256), (3, 2), score_sorted[:6],
                                vmin=vmin, vmax=vmax, cmap=parula)
score_str = "Score:"+", ".join(["%.0f"%s for s in  score_sorted[:6]])
#%%
mtg_frm_init = color_framed_montages(img_col_init, (256, 256), (3, 2), score_init_sorted[:6],
                                vmin=vmin, vmax=vmax, cmap=parula)
score_str_init = "Score:"+", ".join(["%.0f"%s for s in  score_init_sorted[:6]])
#%%
from os.path import join
outdir = r"E:\OneDrive - Harvard University\NeurRep2022_NeurIPS\Figures\Interpretable_LevelSet\src"
plt.imsave(join(outdir, f"{Animal}_{Expi:02d}_EvolRef_top.png"), mtg[0])
plt.imsave(join(outdir, f"{Animal}_{Expi:02d}_EvolRef_top_colorfrm.png"), mtg_frm[0])
plt.imsave(join(outdir, f"{Animal}_{Expi:02d}_EvolRef_top_colorfrm_de-adapt.png"), mtg_frm_init[0])
#%%
def show_img(img, score_str=""):
    fig = plt.figure()
    plt.imshow(img)
    plt.axis("off")
    plt.title(score_str)
    plt.tight_layout()
    plt.show()
    return fig

fig1 = show_img(mtg[0], score_str)
saveallforms(outdir, f"{Animal}_{Expi:02d}_score", figh=fig1, fmts=["png", "pdf"])

fig2 = show_img(mtg_frm[0], score_str)
saveallforms(outdir, f"{Animal}_{Expi:02d}_colorfrm", figh=fig2, fmts=["png", "pdf"])
#%%
fig3 = show_img(mtg_frm_init[0], score_str_init)
saveallforms(outdir, f"{Animal}_{Expi:02d}_colorfrm_deadapt", figh=fig3, fmts=["png", "pdf"])

#%%
plt.imshow(mtg_frm[0])
plt.axis("off")
plt.title(score_str)
plt.tight_layout()
saveallforms(outdir, f"{Animal}_{Expi:02d}_score_colorfrm", fmts=["png", "pdf"])
plt.show()