"""Find RBF approximation of CNN activations
Use optimizations to find the images with a certain activation level while maximizing or minimizing the distance to a target.

"""
import re
import os
import torch
from torch.optim import Adam
import torch.nn.functional as F
import torchvision.models as models
from lpips import LPIPS
import numpy as np
import pandas as pd
from tqdm import tqdm
from os.path import join
import matplotlib.pylab as plt
from easydict import EasyDict
from collections import defaultdict
from scipy.stats import pearsonr, spearmanr
from core.utils.GAN_utils import upconvGAN
from core.utils.CNN_scorers import TorchScorer
from core.utils.Optimizers import CholeskyCMAES
from core.utils.plot_utils import show_imgrid, save_imgrid, save_imgrid_by_row
from core.utils.layer_hook_utils import featureFetcher, get_module_names, register_hook_by_module_names, layername_dict
from core.utils.grad_RF_estim import grad_RF_estimate, gradmap2RF_square, fit_2dgauss, show_gradmap
from core.proto_analysis_lib import sweep_folder, visualize_proto_by_level, visualize_score_imdist, \
        calc_proto_diversity_per_bin, visualize_diversity_by_bin, filter_visualize_codes
from core.latent_explore_lib import latent_explore_batch, latent_diversity_explore, \
    latent_diversity_explore_wRF_fixval, search_peak_evol, search_peak_gradient, calc_rfmap

#%%
def pick_goodimages(S, rfmaptsr, thresh=2.5):
    imdist_good = S.imdist[S.score > thresh]
    imdist_bad = S.imdist[S.score < thresh]
    print(f"imdist good {imdist_good.mean():.2f}+-{imdist_good.std():.2f}\t"
          f"{imdist_bad.mean():.2f}+-{imdist_bad.std():.2f}")

    show_imgrid(G.visualize_batch(S.dz_final[S.score > thresh, :].cuda()).cpu()*rfmaptsr.cpu())


#%% New experiments
Dist = LPIPS(net="squeeze", ).cuda()
Dist.requires_grad_(False)
G = upconvGAN("fc6").cuda()
G.requires_grad_(False)
#%%
scorer = TorchScorer("resnet50")
# scorer.select_unit(("resnet50", ".layer3.Bottleneck5", 5, 7, 7), allow_grad=True)
# scorer.select_unit(("resnet50", ".layer4.Bottleneck2", 5, 3, 3), allow_grad=True)
# scorer.select_unit(("resnet50", ".layer2.Bottleneck3", 5, 13, 13), allow_grad=True)
# scorer.select_unit(("resnet50", ".layer2.Bottleneck3", 10, 13, 13), allow_grad=True)
# scorer.select_unit(("resnet50", ".layer4.Bottleneck0", 5, 3, 3), allow_grad=True)
scorer.select_unit(("resnet50", ".layer4.Bottleneck1", 5, 3, 3), allow_grad=True)
#%%
unitlist = [("resnet50", ".layer4.Bottleneck2", 10, 3, 3),
            ("resnet50", ".layer3.Bottleneck2", 5, 7, 7),
            ("resnet50", ".layer3.Bottleneck0", 5, 7, 7),
            ("resnet50", ".layer3.Bottleneck5", 10, 7, 7),
            ("resnet50", ".layer4.Bottleneck1", 5, 3, 3),
            ("resnet50", ".layer4.Bottleneck0", 10, 3, 3),
            ("resnet50", ".layer2.Bottleneck1", 5, 13, 13),
            ("resnet50", ".layer2.Bottleneck3", 15, 13, 13),
            ]
#%%
unitlist = [#("resnet50_linf8", ".layer4.Bottleneck2", 10, 3, 3),
            #("resnet50_linf8", ".layer4.Bottleneck1", 5, 3, 3),
            # ("resnet50_linf8", ".layer3.Bottleneck5", 5, 7, 7),
            # ("resnet50_linf8", ".layer3.Bottleneck2", 5, 7, 7),
            # ("resnet50_linf8", ".layer2.Bottleneck3", 5, 13, 13),
            # ("resnet50_linf8", ".layer3.Bottleneck0", 5, 7, 7),
            # ("resnet50_linf8", ".layer3.Bottleneck5", 10, 7, 7),
            # ("resnet50_linf8", ".layer4.Bottleneck0", 10, 3, 3),
            # ("resnet50_linf8", ".layer2.Bottleneck1", 5, 13, 13),
            # ("resnet50_linf8", ".layer2.Bottleneck1", 10, 13, 13), # some problem in min
            ("resnet50_linf8", ".layer2.Bottleneck3", 10, 13, 13),
            ("resnet50_linf8", ".layer3.Bottleneck0", 10, 7, 7),
            ("resnet50_linf8", ".layer3.Bottleneck2", 10, 7, 7),
            ("resnet50_linf8", ".layer3.Bottleneck5", 10, 7, 7),
            ("resnet50_linf8", ".layer3.Bottleneck5", 15, 7, 7),
            ("resnet50_linf8", ".layer4.Bottleneck0", 15, 3, 3),
            ("resnet50_linf8", ".layer4.Bottleneck2", 5, 3, 3),
            ("resnet50_linf8", ".layer4.Bottleneck2", 10, 3, 3),
            ]

repn = 4
for unit_tup in unitlist:
    netname = unit_tup[0]
    scorer = TorchScorer(netname)
    scorer.select_unit(unit_tup, allow_grad=True)
    unitlabel = "%s-%d" % (scorer.layer.replace(".Bottleneck", "-Btn").strip("."), scorer.chan)
    outroot = join(r"E:\insilico_exps\proto_diversity", netname)
    outrf_dir = join(outroot, unitlabel+"_rf")
    os.makedirs(outrf_dir, exist_ok=True)
    # Compute RF map for the unit.
    rfmaptsr, rfmapnp, fitdict = calc_rfmap(scorer, outrf_dir, label=unitlabel, use_fit=True, )
    #%%
    # Perform evolution with CMA and gradient. Save the code and image
    z_evol, img_evol, resp_evol, resp_all, z_all = search_peak_evol(G, scorer, nstep=100)
    z_base, img_base, resp_base = search_peak_gradient(G, scorer, z_evol, resp_evol, nstep=100)
    resp_base = torch.tensor(resp_base).float().cuda()
    save_imgrid(img_base, join(outrf_dir, "proto_peak.png"))
    save_imgrid(img_base*rfmaptsr, join(outrf_dir, "proto_peak_rf.png"))
    torch.save(dict(z_base=z_base, img_base=img_base, resp_base=resp_base,
                    z_evol=z_evol, img_evol=img_evol, resp_evol=resp_evol,
                    unit_tuple=unit_tup, unitlabel=unitlabel),
               join(outrf_dir, "proto_optim.pt"))
    #%%
    for ratio in np.arange(0.10, 1.1, 0.1):
        trial_dir = join(outrf_dir, "fix%s_max"%(("%.2f"%ratio).replace(".", "")))
        opts = dict(dz_sigma=1.5, imgdist_obj="max", alpha_img=5.0,
                    score_obj="fix", score_fixval=resp_base * ratio, alpha_score=1.0,
                    noise_std=0.3, steps=150)
        S = latent_explore_batch(G, Dist, scorer, z_base, resp_base, img_base, rfmaptsr, opts, trial_dir, repn=repn)
        S_sel = filter_visualize_codes(trial_dir, thresh=resp_base.item() * ratio, abinit=False,
                                       err=resp_base.item() * ratio * 0.2)

    for ratio in np.arange(0.10, 1.1, 0.1):
        trial_dir = join(outrf_dir, "fix%s_max_abinit"%(("%.2f"%ratio).replace(".", "")))
        opts = dict(dz_sigma=3, imgdist_obj="max", alpha_img=5.0,
                    score_obj="fix", score_fixval=resp_base * ratio, alpha_score=1.0,
                    noise_std=0.3, steps=150)
        S = latent_explore_batch(G, Dist, scorer, torch.zeros_like(z_base), resp_base, img_base, rfmaptsr, opts, trial_dir, repn=repn)
        S_sel = filter_visualize_codes(trial_dir, thresh=resp_base.item() * ratio, abinit=True,
                                       err=resp_base.item() * ratio * 0.2)

    for ratio in np.arange(0.10, 1.1, 0.1):
        trial_dir = join(outrf_dir, "fix%s_min"%(("%.2f"%ratio).replace(".", "")))
        opts = dict(dz_sigma=1.5, imgdist_obj="min", alpha_img=5.0,
                    score_obj="fix", score_fixval=resp_base * ratio, alpha_score=1.0,
                    noise_std=0.3, steps=150)
        S = latent_explore_batch(G, Dist, scorer, z_base, resp_base, img_base, rfmaptsr, opts, trial_dir, repn=repn)
        S_sel = filter_visualize_codes(trial_dir, thresh=resp_base.item() * ratio, abinit=False,
                                       err=resp_base.item() * 0.1) #ratio * 0.2

    for suffix in ["min", "max", "max_abinit"]:
        sumdict, sumdir = sweep_folder(outrf_dir, dirnm_pattern=f"fix.*_{suffix}$",
                                       sum_sfx=f"summary_{suffix}")
        visualize_proto_by_level(G, sumdict, sumdir, bin_width=0.10, relwidth=0.25, )
        visualize_score_imdist(sumdict, sumdir, )
        df, pixdist_dict, lpipsdist_dict, lpipsdistmat_dict = calc_proto_diversity_per_bin(G, Dist, sumdict, sumdir,
                                                                   bin_width=0.10, distsampleN=40)
        visualize_diversity_by_bin(df, sumdir)


#%% Development zone
#%%
feattsr_all = []
resp_all = []
z_all = []
optimizer = CholeskyCMAES(4096, population_size=None, init_sigma=3.0)
z_arr = np.zeros((1, 4096))  # optimizer.init_x
pbar = tqdm(range(100))
for i in pbar:
    imgs = G.visualize(torch.tensor(z_arr).float().cuda())
    resp = scorer.score(imgs, )
    z_arr_new = optimizer.step_simple(resp, z_arr)
    z_arr = z_arr_new
    resp_all.append(resp)
    z_all.append(z_arr)
    print(f"{i}: {resp.mean():.2f}+-{resp.std():.2f}")
    # with torch.no_grad():
    #     featnet(scorer.preprocess(imgs, input_scale=1.0))
    #
    # del imgs
    # pbar.set_description(f"{i}: {resp.mean():.2f}+-{resp.std():.2f}")
    # feattsr = featFetcher[regresslayer][:, :, 6, 6]
    # feattsr_all.append(feattsr.cpu().numpy())

resp_all = np.concatenate(resp_all, axis=0)
z_all = np.concatenate(z_all, axis=0)
# feattsr_all = np.concatenate(feattsr_all, axis=0)

z_base = torch.tensor(z_all.mean(axis=0, keepdims=True)).float().cuda()
img_base = G.visualize(z_base)
resp_base = scorer.score_tsr_wgrad(img_base, )
#%% Maximize invariance / Diversity
#%%
scorer = TorchScorer("resnet50")
scorer.select_unit(("resnet50", ".layer3.Bottleneck5", 5, 6, 6), allow_grad=True)
#%%
dz = 0.1 * torch.randn(1, 4096).cuda()
dz.requires_grad_()
optimizer = Adam([dz], lr=0.1)
for i in tqdm(range(200)):
    optimizer.zero_grad()
    curimg = G.visualize(z_base + dz)
    resp_new = scorer.score_tsr_wgrad(curimg, )
    score_loss = (resp_base - resp_new)
    img_dist = Dist(img_base, curimg)
    loss = score_loss
    loss.backward()
    optimizer.step()
    print(f"{i}: {loss.item():.2f} new score {resp_new.item():.2f} "
          f"old score {resp_base.item():.2f} img dist{img_dist.item():.2f}")
# show_imgrid(curimgs)
#%%
z_base = z_base + dz.detach().clone()
z_base.detach_()
#%%
img_base = G.visualize(z_base)
resp_base = scorer.score_tsr_wgrad(img_base, )
#%%
out_dir = r"E:\insilico_exps\proto_diversity\resnet50_linf8\layer3-Btn5-5"
opts = dict(alpha=10.0, dz_sigma=2.0, batch_size=5, steps=150, lr=0.1, )
dz_init_col = []
dz_col = []
score_col = []
imdist_col = []
for i in range(100):
    dzs_init, dzs, img_dists, curimgs, scores = latent_diversity_explore(G, Dist, scorer, z_base,
                  **opts)
    save_imgrid(curimgs, join(out_dir, f"proto_divers_{i}.png"))
    dz_init_col.append(dzs_init)
    dz_col.append(dzs)
    score_col.append(scores)
    imdist_col.append(img_dists)

dz_init_tsr = torch.cat(dz_init_col, dim=0)
dz_final_tsr = torch.cat(dz_col, dim=0)
score_vec = torch.cat(score_col, dim=0)
imdist_vec = torch.cat(imdist_col, dim=0)
torch.save({"dz_init": dz_init_tsr, "dz_final": dz_final_tsr, "score": score_vec, "imdist": imdist_vec,
            "z_base": z_base, "score_base": resp_base, "opts": opts},
           join(out_dir, "diversity_dz_score.pt"))
#%% Calculate RF mask of the unit.
from core.utils.grad_RF_estim import grad_RF_estimate, gradmap2RF_square, fit_2dgauss, show_gradmap
cent_pos = (6, 6)
outrf_dir = r"E:\insilico_exps\proto_diversity\resnet50_linf8\layer3-Btn5-5_rf"
gradAmpmap = grad_RF_estimate(scorer.model, scorer.layer, (slice(None), cent_pos[0], cent_pos[1]),
                          input_size=(3, 227, 227), device="cuda", show=False, reps=200, batch=4)
show_gradmap(gradAmpmap, )
fitdict = fit_2dgauss(gradAmpmap, "layer3-Btn5-5", outdir=outrf_dir, plot=True)
#%%
rfmap = fitdict.fitmap
rfmap = fitdict.gradAmpmap
rfmap /= rfmap.max()
rfmaptsr = torch.from_numpy(rfmap).float().cuda().unsqueeze(0).unsqueeze(0)
rfmaptsr = torch.nn.functional.interpolate(rfmaptsr,
               (256, 256), mode="bilinear", align_corners=True)
rfmap_full = rfmaptsr.cpu()[0,0,:].unsqueeze(2).numpy()


#%%
# out_dir_rf = r"E:\insilico_exps\proto_diversity\resnet50_linf8\layer3-Btn5-5_rf"
# alpha=10.0; dz_sigma=2.0
out_dir_rf = r"E:\insilico_exps\proto_diversity\resnet50_linf8\layer3-Btn5-5_rf_far200"
alpha = 200.0; dz_sigma = 3.0
dz_init_col = []
dz_col = []
score_col = []
imdist_col = []
for i in range(100):
    dzs_init, dzs, img_dists, curimgs, scores = latent_diversity_explore_wRF(G, Dist, scorer, z_base,
                  rfmaptsr, alpha=alpha, dz_sigma=dz_sigma, batch_size=5, steps=150, lr=0.1, )
    save_imgrid(curimgs, join(out_dir_rf, f"proto_divers_{i}.png"))
    save_imgrid(curimgs*rfmaptsr.cpu(), join(out_dir_rf, f"proto_divers_wRF_{i}.png"))
    dz_init_col.append(dzs_init)
    dz_col.append(dzs)
    score_col.append(scores)
    imdist_col.append(img_dists)

dz_init_tsr = torch.cat(dz_init_col, dim=0)
dz_final_tsr = torch.cat(dz_col, dim=0)
score_vec = torch.cat(score_col, dim=0)
imdist_vec = torch.cat(imdist_col, dim=0)
torch.save({"dz_init": dz_init_tsr, "dz_final": dz_final_tsr, "score": score_vec, "imdist": imdist_vec,
            "z_base": z_base, "score_base": resp_base, "alpha": alpha, "dz_sigma": dz_sigma},
           join(out_dir_rf, "diversity_dz_score.pt"))
# 4.54
#%%
out_dir_rf = r"E:\insilico_exps\proto_diversity\resnet50_linf8\layer3-Btn5-5_rf_fix05_min"
os.makedirs(out_dir_rf, exist_ok=True)
opts = dict(dz_sigma=1.0, score_obj="fix", score_fixval=resp_base * 0.5, alpha_score=1.0,
        imgdist_obj="min", alpha_img=0.2)
dz_init_col = []
dz_col = []
score_col = []
imdist_col = []
for i in range(5):
    dzs_init, dzs, img_dists, curimgs, scores = latent_diversity_explore_wRF_fixval(G, Dist, scorer,
                        z_base, rfmaptsr, **opts)
    save_imgrid(curimgs, join(out_dir_rf, f"proto_divers_{i}.png"))
    save_imgrid(curimgs*rfmaptsr.cpu(), join(out_dir_rf, f"proto_divers_wRF_{i}.png"))
    dz_init_col.append(dzs_init)
    dz_col.append(dzs)
    score_col.append(scores)
    imdist_col.append(img_dists)

dz_init_tsr = torch.cat(dz_init_col, dim=0)
dz_final_tsr = torch.cat(dz_col, dim=0)
score_vec = torch.cat(score_col, dim=0)
imdist_vec = torch.cat(imdist_col, dim=0)
torch.save({"dz_init": dz_init_tsr, "dz_final": dz_final_tsr, "score": score_vec, "imdist": imdist_vec,
            "z_base": z_base, "score_base": resp_base, "rfmaptsr": rfmaptsr, "opts": opts, },
           join(out_dir_rf, "diversity_dz_score.pt"))

#%% New interface
#%% ab initio generation of images matching
out_dir_rf = join(outroot, "layer3-Btn5-5_rf_fix05_min")
opts = dict(dz_sigma=3.0, score_obj="fix", score_fixval=resp_base * 0.5, alpha_score=1.0,
            imgdist_obj="min", alpha_img=0.2)
S = latent_explore_batch(G, Dist, scorer, z_base, resp_base, img_base, rfmaptsr, opts, out_dir_rf, repn=20)
#%%
out_dir_rf = join(outroot, "layer3-Btn5-5_rf_fix05_min_abinit")
opts = dict(dz_sigma=3.0, imgdist_obj="min", alpha_img=0.2,
            score_obj="fix", score_fixval=resp_base * 0.5, alpha_score=1.0,)
S = latent_explore_batch(G, Dist, scorer, torch.zeros_like(z_base), resp_base, img_base, rfmaptsr, opts, out_dir_rf, repn=20)
#%%
out_dir_rf = join(outroot, "layer3-Btn5-5_rf_fix05_max")
opts = dict(dz_sigma=3.0, imgdist_obj="max", alpha_img=0.1,
            score_obj="fix", score_fixval=resp_base * 0.5, alpha_score=1.0, )
S = latent_explore_batch(G, Dist, scorer, z_base, resp_base, img_base, rfmaptsr, opts, out_dir_rf, repn=20)
#%%
out_dir_rf = join(outroot, "layer3-Btn5-5_rf_fix05_max_abinit")
opts = dict(dz_sigma=3.0, imgdist_obj="max", alpha_img=0.05,
            score_obj="fix", score_fixval=resp_base * 0.5, alpha_score=1.0, )
S = latent_explore_batch(G, Dist, scorer, torch.zeros_like(z_base), resp_base, img_base, rfmaptsr, opts, out_dir_rf, repn=20)
S_sel = filter_visualize_codes(out_dir_rf, thresh=2.7, )

#%%
out_dir_rf = join(outroot, "layer3-Btn5-5_rf_fix08_max_abinit")
opts = dict(dz_sigma=3.0, imgdist_obj="max", alpha_img=0.05,
            score_obj="fix", score_fixval=resp_base * 0.8, alpha_score=1.0, )
S = latent_explore_batch(G, Dist, scorer, torch.zeros_like(z_base), resp_base, img_base, rfmaptsr, opts, out_dir_rf, repn=20)
S_sel = filter_visualize_codes(out_dir_rf, thresh=4.4, )
#%%
out_dir_rf = join(outroot, "layer3-Btn5-5_rf_fix08_max")
opts = dict(dz_sigma=3.0, imgdist_obj="max", alpha_img=5.00,
            score_obj="fix", score_fixval=resp_base * 0.8, alpha_score=1.0, )
S = latent_explore_batch(G, Dist, scorer, z_base, resp_base, img_base, rfmaptsr, opts, out_dir_rf, repn=20)
S_sel = filter_visualize_codes(out_dir_rf, thresh=4.4, abinit=False)
#%%
#%%
out_dir_rf = join(outroot, "layer3-Btn5-5_rf_fix03_max")
opts = dict(dz_sigma=3.0, imgdist_obj="max", alpha_img=5.00,
            score_obj="fix", score_fixval=resp_base * 0.3, alpha_score=1.0, )
S = latent_explore_batch(G, Dist, scorer, z_base, resp_base, img_base, rfmaptsr, opts, out_dir_rf, repn=20)
S_sel = filter_visualize_codes(out_dir_rf, thresh=1.4, abinit=False)

#%%
out_dir_rf = join(outroot, "layer3-Btn5-5_rf_fix015_max")
opts = dict(dz_sigma=1.5, imgdist_obj="max", alpha_img=5.00,
            score_obj="fix", score_fixval=resp_base * 0.15, alpha_score=1.0, )
S = latent_explore_batch(G, Dist, scorer, z_base, resp_base, img_base, rfmaptsr, opts, out_dir_rf, repn=20)
S_sel = filter_visualize_codes(out_dir_rf, thresh=0.65, abinit=False)
#%%
for ratio in np.arange(0.10, 1, 0.05):
    out_dir_rf = join(outroot, "layer3-Btn5-5_rf_fix%s_max"%(("%.2f"%ratio).replace(".", "")))
    opts = dict(dz_sigma=1.5, imgdist_obj="max", alpha_img=5.0,
                score_obj="fix", score_fixval=resp_base * ratio, alpha_score=1.0, )
    S = latent_explore_batch(G, Dist, scorer, z_base, resp_base, img_base, rfmaptsr, opts, out_dir_rf, repn=40)
    S_sel = filter_visualize_codes(out_dir_rf, thresh=resp_base.item() * ratio, abinit=False,
                                   err=resp_base.item() * ratio * 0.2)
#%
for ratio in np.arange(0.05, 1, 0.05):
    out_dir_rf = join(outroot, "layer3-Btn5-5_rf_fix%s_max_abinit"%(("%.2f"%ratio).replace(".", "")))
    opts = dict(dz_sigma=3, imgdist_obj="max", alpha_img=5.0,
                score_obj="fix", score_fixval=resp_base * ratio, alpha_score=1.0, )
    S = latent_explore_batch(G, Dist, scorer, torch.zeros_like(z_base), resp_base, img_base, rfmaptsr, opts, out_dir_rf, repn=40)
    S_sel = filter_visualize_codes(out_dir_rf, thresh=resp_base.item() * ratio, abinit=True,
                                   err=resp_base.item() * ratio * 0.2)
#%%
ratio = 0.5
out_dir_rf = join(outroot, "layer3-Btn5-5_rf_fix%s_min"%(("%.2f"%ratio).replace(".", "")))
opts = dict(dz_sigma=2, imgdist_obj="min", alpha_img=5.0,
            score_obj="fix", score_fixval=resp_base * ratio, alpha_score=1.0, )
S = latent_explore_batch(G, Dist, scorer, z_base, resp_base, img_base, rfmaptsr, opts, out_dir_rf, repn=40)
S_sel = filter_visualize_codes(out_dir_rf, thresh=resp_base.item() * ratio, abinit=True,
                               err=resp_base.item() * ratio * 0.2)
ratio = 0.5
out_dir_rf = join(outroot, "layer3-Btn5-5_rf_fix%s_min_abinit"%(("%.2f"%ratio).replace(".", "")))
opts = dict(dz_sigma=2, imgdist_obj="min", alpha_img=5.0,
            score_obj="fix", score_fixval=resp_base * ratio, alpha_score=1.0, )
S = latent_explore_batch(G, Dist, scorer, torch.zeros_like(z_base), resp_base, img_base, rfmaptsr, opts, out_dir_rf, repn=40)
S_sel = filter_visualize_codes(out_dir_rf, thresh=resp_base.item() * ratio, abinit=True,
                               err=resp_base.item() * ratio * 0.2)
#%%


#%%
S_sel = filter_visualize_codes(join(outroot, "layer3-Btn5-5_rf_fix05_min_abinit"),
                               thresh=2.4, abinit=True,)
#%%
S_sel = filter_visualize_codes(join(outroot, "layer3-Btn5-5_rf_fix05_min"),
                               thresh=2.4, abinit=False,)
# pick_goodimages(S, rfmaptsr, thresh=2.5)
#%% Copy to a summary folder
import shutil
abinit = True
os.makedirs(join(outroot, "summary"), exist_ok=True)
for ratio in np.arange(0.05, 1, 0.05):
    ratio_str = ("%.2f"%ratio).replace(".", "")
    out_dir_rf = join(outroot, "layer3-Btn5-5_rf_fix%s_max%s"%
                      (ratio_str, "_abinit" if abinit else ""))
    outfn = "%sproto_divers_%s.png"%("abinit_" if abinit else "", ratio_str)
    shutil.copy2(join(out_dir_rf, "sorted", "proto_divers_wRF_0.png"),
                 join(outroot, "summary", outfn))



#%%
imdist_good = imdist_vec[score_vec > 2.8]
imdist_bad = imdist_vec[score_vec < 2.8]
print(f"imdist good {imdist_good.mean():.2f}+-{imdist_good.std():.2f}\t"
      f"{imdist_bad.mean():.2f}+-{imdist_bad.std():.2f}")
#%%
show_imgrid(G.visualize(dz_final_tsr[score_vec > 2.7, :].cuda()).cpu()*rfmaptsr.cpu())
#%%
cos_angle_good = (dz_final_tsr[score_vec>0.05,:] @ z_base.cpu().T) \
    / dz_final_tsr[score_vec>0.05,:].norm(dim=1, keepdim=True) / z_base.cpu().norm(dim=1)
cos_angle_bad = (dz_final_tsr[score_vec<0.05,:] @ z_base.cpu().T) \
    / dz_final_tsr[score_vec<0.05,:].norm(dim=1, keepdim=True) / z_base.cpu().norm(dim=1)
#%%
figh, axs = plt.subplots(1, 2, figsize=(10, 5))
axs[0].imshow(img_base.cpu().detach().numpy()[0].transpose(1, 2, 0))
axs[1].imshow(curimg.cpu().detach().numpy()[0].transpose(1, 2, 0))
plt.show()
#%%
figh, axs = plt.subplots(1, 2, figsize=(10, 5))
axs[0].imshow(img_base.cpu().detach().numpy()[0].transpose(1, 2, 0) * rfmap_full)
axs[1].imshow(curimg.cpu().detach().numpy()[0].transpose(1, 2, 0) * rfmap_full)
plt.show()
#%%
show_imgrid(curimgs * rfmaptsr.cpu())
#%%
#%% New iteration
z_base = z_base + dz.detach()
img_base = G.visualize(z_base)
resp_base = scorer.score_tsr_wgrad(img_base, )
#%% Dev Zone
#%% Single thread
dz = 0.1 * torch.randn(1, 4096).cuda()
dz.requires_grad_()
optimizer = Adam([dz], lr=0.05)
for i in tqdm(range(100)):
    optimizer.zero_grad()
    curimg = G.visualize(z_base + dz)
    resp_new = scorer.score_tsr_wgrad(curimg, )
    score_loss = (resp_base - resp_new)
    img_dist = Dist(img_base, curimg)
    loss = score_loss - img_dist
    loss.backward()
    optimizer.step()
    print(f"{i}: {loss.item():.2f} new score {resp_new.item():.2f} "
          f"old score {resp_base.item():.2f} img dist{img_dist.item():.2f}")
# %% Multi thread exploration
alpha = 10.0
dz_sigma = 3  # 0.4
dzs = dz_sigma * torch.randn(5, 4096).cuda()
dzs.requires_grad_()
optimizer = Adam([dzs], lr=0.1)
for i in tqdm(range(150)):
    optimizer.zero_grad()
    curimgs = G.visualize(z_base + dzs)
    resp_news = scorer.score_tsr_wgrad(curimgs, )
    score_loss = (resp_base - resp_news)
    resp_news_mid = scorer.score_tsr_wgrad(G.visualize(z_base + dzs / 2), )
    score_mid_loss = (resp_base - resp_news_mid)
    img_dists = Dist(img_base, curimgs)
    loss = (score_loss + score_mid_loss - alpha * img_dists).mean()
    loss.backward()
    optimizer.step()
    print(f"{i}: {loss.item():.2f} new score {resp_news.mean().item():.2f} mid score {resp_news_mid.mean().item():.2f} "
          f"old score {resp_base.item():.2f} img dist{img_dists.mean().item():.2f}")
#%%


#%% md
# Trying different spatial weighted Distance computation
#%%
Dist.spatial = False
Dimg_origin = Dist(img_base, curimgs.cuda())
#%% Mask the image before computing the distance
Dist.spatial = False
dist_mask = Dist(img_base * rfmaptsr, curimgs.cuda() * rfmaptsr)
#%% with spatial weighting of feature distance
Dist.spatial = True
distmaps = Dist(img_base, curimgs.cuda())
Dimg_weighted = (distmaps * rfmaptsr).sum(dim=[1,2,3]) / rfmaptsr.sum(dim=[1,2,3])
#%%