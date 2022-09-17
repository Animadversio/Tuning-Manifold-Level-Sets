import re
import os
import torch
from lpips import LPIPS
import numpy as np
from os.path import join
import matplotlib.pylab as plt
from easydict import EasyDict
from collections import defaultdict
from scipy.stats import pearsonr, spearmanr
from core.utils.GAN_utils import upconvGAN
from core.utils.CNN_scorers import TorchScorer
from core.utils.plot_utils import show_imgrid, save_imgrid, save_imgrid_by_row
from core.proto_analysis_lib import sweep_folder, visualize_proto_by_level, visualize_score_imdist, \
        calc_proto_diversity_per_bin, visualize_diversity_by_bin, filter_visualize_codes
from core.latent_explore_lib import latent_explore_batch, latent_diversity_explore, \
    latent_diversity_explore_wRF_fixval, search_peak_evol, search_peak_gradient, calc_rfmap

#%% New experiments
Dist = LPIPS(net="squeeze", ).cuda()
Dist.requires_grad_(False)
G = upconvGAN("fc6").cuda()
G.requires_grad_(False)
#%%
# unitlist = [#("resnet50_linf8", ".layer4.Bottleneck2", 10, 3, 3),
#             #("resnet50_linf8", ".layer4.Bottleneck1", 5, 3, 3),
#             # ("resnet50_linf8", ".layer3.Bottleneck5", 5, 7, 7),
#             # ("resnet50_linf8", ".layer3.Bottleneck2", 5, 7, 7),
#             # ("resnet50_linf8", ".layer2.Bottleneck3", 5, 13, 13),
#             # ("resnet50_linf8", ".layer3.Bottleneck0", 5, 7, 7),
#             # ("resnet50_linf8", ".layer3.Bottleneck5", 10, 7, 7),
#             # ("resnet50_linf8", ".layer4.Bottleneck0", 10, 3, 3),
#             # ("resnet50_linf8", ".layer2.Bottleneck1", 5, 13, 13),
#             # ("resnet50_linf8", ".layer2.Bottleneck1", 10, 13, 13), # some problem in min
#             ("resnet50_linf8", ".layer2.Bottleneck3", 10, 13, 13),
#             ("resnet50_linf8", ".layer3.Bottleneck0", 10, 7, 7),
#             ("resnet50_linf8", ".layer3.Bottleneck2", 10, 7, 7),
#             ("resnet50_linf8", ".layer3.Bottleneck5", 10, 7, 7),
#             ("resnet50_linf8", ".layer3.Bottleneck5", 15, 7, 7),
#             ("resnet50_linf8", ".layer4.Bottleneck0", 15, 3, 3),
#             ("resnet50_linf8", ".layer4.Bottleneck2", 5, 3, 3),
#             ("resnet50_linf8", ".layer4.Bottleneck2", 10, 3, 3),
#             ]
unitlist = [("resnet50_linf8", ".layer4.Bottleneck2", 30, 3, 3),
            ("resnet50_linf8", ".layer4.Bottleneck0", 30, 3, 3),
            ("resnet50_linf8", ".layer3.Bottleneck5", 30, 7, 7),
            ("resnet50_linf8", ".layer3.Bottleneck2", 30, 7, 7),
            ("resnet50_linf8", ".layer2.Bottleneck3", 30, 13, 13),
            ("resnet50_linf8", ".layer1.Bottleneck2", 30, 28, 28), ] #
repn = 1
batch_size = 40
for unit_chan in range(30, 40):
    for unit_tup in unitlist:
        netname = unit_tup[0]
        unit_tup_new = (unit_tup[0], unit_tup[1], unit_chan, unit_tup[3], unit_tup[4])

        scorer = TorchScorer(netname)
        scorer.select_unit(unit_tup_new, allow_grad=True)
        unitlabel = "%s-%d" % (scorer.layer.replace(".Bottleneck", "-Btn").strip("."), scorer.chan)
        outroot = join(r"/home/binxuwang/insilico_exp/proto_diversity", netname)
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
        for ratio in np.arange(0.0, 1.1, 0.1):
            trial_dir = join(outrf_dir, "fix%s_max"%(("%.2f"%ratio).replace(".", "")))
            opts = dict(imgdist_obj="max", alpha_img=5.0,
                        score_obj="fix", score_fixval=resp_base * ratio, alpha_score=1.0,
                        dz_sigma=1.5, noise_std=0.3, steps=75, batch_size=batch_size)
            S = latent_explore_batch(G, Dist, scorer, z_base, resp_base, img_base, rfmaptsr, opts, trial_dir, repn=repn)
            S_sel = filter_visualize_codes(G, trial_dir, thresh=resp_base.item() * ratio, abinit=False,
                                           err=resp_base.item() * 0.1)

        for ratio in np.arange(0.0, 1.1, 0.1):
            trial_dir = join(outrf_dir, "fix%s_min"%(("%.2f"%ratio).replace(".", "")))
            opts = dict(imgdist_obj="min", alpha_img=5.0,
                        score_obj="fix", score_fixval=resp_base * ratio, alpha_score=1.0,
                        dz_sigma=1.5, noise_std=0.3, steps=75, batch_size=batch_size)
            S = latent_explore_batch(G, Dist, scorer, z_base, resp_base, img_base, rfmaptsr, opts, trial_dir, repn=repn)
            S_sel = filter_visualize_codes(G, trial_dir, thresh=resp_base.item() * ratio, abinit=False,
                                           err=resp_base.item() * 0.1) #ratio * 0.2

        for ratio in np.arange(0.0, 1.1, 0.1):
            trial_dir = join(outrf_dir, "fix%s_max_abinit"%(("%.2f"%ratio).replace(".", "")))
            opts = dict(imgdist_obj="max", alpha_img=5.0,
                        score_obj="fix", score_fixval=resp_base * ratio, alpha_score=1.0,
                        dz_sigma=3, noise_std=0.3, steps=75, batch_size=batch_size)
            S = latent_explore_batch(G, Dist, scorer, torch.zeros_like(z_base), resp_base, img_base, rfmaptsr, opts, trial_dir, repn=repn)
            S_sel = filter_visualize_codes(G, trial_dir, thresh=resp_base.item() * ratio, abinit=True,
                                           err=resp_base.item() * 0.1)

        for ratio in np.arange(0.0, 1.1, 0.1):
            trial_dir = join(outrf_dir, "fix%s_none_abinit"%(("%.2f"%ratio).replace(".", "")))
            opts = dict(imgdist_obj="none", alpha_img=0.0,
                        score_obj="fix", score_fixval=resp_base * ratio, alpha_score=1.0,
                        dz_sigma=3, noise_std=0.3, steps=75, batch_size=batch_size)
            S = latent_explore_batch(G, Dist, scorer, torch.zeros_like(z_base), resp_base, img_base, rfmaptsr, opts, trial_dir, repn=repn)
            S_sel = filter_visualize_codes(G, trial_dir, thresh=resp_base.item() * ratio, abinit=True,
                                           err=resp_base.item() * 0.1)

        for suffix in ["min", "max", "max_abinit", "none_abinit"]:
            sumdict, sumdir = sweep_folder(outrf_dir, dirnm_pattern=f"fix.*_{suffix}$",
                                           sum_sfx=f"summary_{suffix}")
            visualize_proto_by_level(G, sumdict, sumdir, bin_width=0.10, relwidth=0.25, )
            visualize_score_imdist(sumdict, sumdir, )
            df, pixdist_dict, lpipsdist_dict, lpipsdistmat_dict = calc_proto_diversity_per_bin(G, Dist, sumdict, sumdir,
                                                                       bin_width=0.10, distsampleN=40)
            visualize_diversity_by_bin(df, sumdir)
