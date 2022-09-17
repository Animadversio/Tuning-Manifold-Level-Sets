import re
import os
import torch
from torch.optim import Adam
import torch.nn.functional as F
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
from core.utils.grad_RF_estim import grad_RF_estimate, gradmap2RF_square, fit_2dgauss, show_gradmap
from core.utils.layer_hook_utils import featureFetcher, get_module_names, register_hook_by_module_names, layername_dict


def latent_diversity_explore(G, Dist, scorer, z_base, resp_base, img_base, dzs=None, alpha=10.0, dz_sigma=3.0,
                      batch_size=5, steps=150, lr=0.1, midpoint=True):
    if dzs is None:
        dzs = dz_sigma * torch.randn(batch_size, 4096).cuda()
    dzs_init = dzs.clone().cpu()
    dzs.requires_grad_()
    optimizer = Adam([dzs], lr=lr)
    for i in tqdm(range(steps)):
        optimizer.zero_grad()
        curimgs = G.visualize(z_base + dzs)
        resp_news = scorer.score_tsr_wgrad(curimgs, )
        score_loss = (resp_base - resp_news)
        if midpoint:
            resp_news_mid = scorer.score_tsr_wgrad(G.visualize(z_base + dzs / 2), )
            score_mid_loss = (resp_base - resp_news_mid)
        else:
            score_mid_loss = torch.zeros_like(score_loss)
        img_dists = Dist(img_base, curimgs)[:, 0, 0, 0]
        loss = (score_loss + score_mid_loss - alpha * img_dists).mean()
        loss.backward()
        optimizer.step()
        print(
            f"{i}: {loss.item():.2f} new score {resp_news.mean().item():.2f} mid score {resp_news_mid.mean().item():.2f} "
            f"old score {resp_base.item():.2f} img dist{img_dists.mean().item():.2f}")
    return dzs_init, dzs.detach().cpu(), img_dists.detach().cpu(),\
           curimgs.detach().cpu(), resp_news.detach().cpu()


def latent_diversity_explore_wRF(G, Dist, scorer, z_base, resp_base, img_base, rfmaptsr, dzs=None, alpha=10.0, dz_sigma=3.0,
                      batch_size=5, steps=150, lr=0.1, midpoint=True):
    """

    :param G:
    :param scorer:
    :param z_base: base latent vector
    :param rfmaptsr: We assume its shape is (1, 1, 256, 256), and its values are in [0, 1]
    :param alpha: The weight of the distance term VS the score term
    :param dz_sigma: The initial std of dz.
    :param dzs: The initial dz.
            If None, it will be sampled from a Gaussian distribution with std dz_sigma.
    :param batch_size:
    :param steps:
    :param lr:
    :param midpoint: If True, the activation of midpoint is also computed.
    :return:
    """
    Dist.spatial = False
    if dzs is None:
        dzs = dz_sigma * torch.randn(batch_size, 4096).cuda()
    dzs_init = dzs.clone().cpu()
    dzs.requires_grad_()
    optimizer = Adam([dzs], lr=lr)
    for i in tqdm(range(steps)):
        optimizer.zero_grad()
        curimgs = G.visualize(z_base + dzs)
        resp_news = scorer.score_tsr_wgrad(curimgs, )
        score_loss = (resp_base - resp_news)
        if midpoint:
            resp_news_mid = scorer.score_tsr_wgrad(G.visualize(z_base + dzs / 2), )
            score_mid_loss = (resp_base - resp_news_mid)
        else:
            score_mid_loss = torch.zeros_like(score_loss)
        img_dists = Dist(img_base * rfmaptsr, curimgs * rfmaptsr)[:, 0, 0, 0]
        loss = (score_loss + score_mid_loss - alpha * img_dists).mean()
        loss.backward()
        optimizer.step()
        print(
            f"{i}: {loss.item():.2f} new score {resp_news.mean().item():.2f} mid score {resp_news_mid.mean().item():.2f} "
            f"old score {resp_base.item():.2f} RF img dist{img_dists.mean().item():.2f}")
    return dzs_init, dzs.detach().cpu(), img_dists.detach().cpu(), \
           curimgs.detach().cpu(), resp_news.detach().cpu()


def latent_diversity_explore_wRF_fixval(G, Dist, scorer, z_base, resp_base, img_base, rfmaptsr, dzs=None, dz_sigma=3.0,
                                        imgdist_obj="max", imgdist_fixval=None, alpha_img=1.0,
                                        score_obj="max", score_fixval=None, alpha_score=1.0,
                                        batch_size=5, steps=150, lr=0.1, noise_std=0.3, ):
    """ Latest version of latent diversity explore.
    Setting 1, fix the score at half maximum. Maximize image distance to prototype
        latent_diversity_explore_wRF_fixval(G, scorer, z_base, rfmaptsr,
                        imgdist_obj="max", score_obj="fix", score_fixval=score_base * 0.5, alpha_score=10.0)
    Setting 2, fix the image distance to prototype. minimize or maximize score
        latent_diversity_explore_wRF_fixval(G, scorer, z_base, rfmaptsr,
                        imgdist_obj="fix", imgdist_fixval=0.1, score_obj="max", alpha_img=10.0)
    :param G:
    :param scorer:
    :param z_base: base latent vector
    :param rfmaptsr: We assume its shape is (1, 1, 256, 256), and its values are in [0, 1]
    :param alpha: The weight of the distance term VS the score term
    :param dz_sigma: The initial std of dz.
    :param dzs: The initial dz.
            If None, it will be sampled from a Gaussian distribution with std dz_sigma.
    :param batch_size:
    :param steps:
    :param lr:
    :param midpoint: If True, the activation of midpoint is also computed.
    :return:
    """
    Dist.spatial = False # scalar output from Dist
    if dzs is None:
        dzs = dz_sigma * torch.randn(batch_size, 4096).cuda()
    dzs_init = dzs.clone().cpu()
    dzs.requires_grad_()
    optimizer = Adam([dzs], lr=lr)
    for i in tqdm(range(steps)):
        optimizer.zero_grad()
        curimgs = G.visualize(z_base + dzs)
        resp_news = scorer.score_tsr_wgrad(curimgs, )
        if score_obj is "max":
            score_loss = (resp_base - resp_news)
        elif score_obj is "min":
            score_loss = - (resp_base - resp_news)
        elif score_obj is "fix":
            score_loss = torch.abs(resp_news - score_fixval)
        else:
            raise ValueError(f"Unknown score_obj {score_obj}")
        # compute distance of images under mask
        if imgdist_obj is "none" or imgdist_obj is None:
            dist_loss = 0.0
        else:
            img_dists = Dist(img_base * rfmaptsr, curimgs * rfmaptsr)[:, 0, 0, 0]
            if imgdist_obj is "max":
                dist_loss = - img_dists
            elif imgdist_obj is "min":
                dist_loss = img_dists
            elif imgdist_obj is "fix":
                dist_loss = torch.abs(img_dists - imgdist_fixval)
            else:
                raise ValueError(f"Unknown imgdist_obj {imgdist_obj}")
        # if midpoint:
        #     resp_news_mid = scorer.score_tsr_wgrad(G.visualize(z_base + dzs / 2), )
        #     score_mid_loss = (resp_base - resp_news_mid)
        # else:
        #     score_mid_loss = torch.zeros_like(score_loss)
        failmask = torch.isclose(resp_news.detach(), torch.zeros(1, device="cuda"))
        loss = (alpha_score * score_loss + alpha_img * dist_loss * (~failmask)).mean()
        # loss = (alpha_score * score_loss + alpha_img * dist_loss).mean()
        loss.backward()
        optimizer.step()
        if failmask.any():
            # add gaussian noise perturbation for the failed evolutions.
            dzs.data = dzs.data + noise_std * \
                torch.randn(batch_size, 4096, device="cuda") * failmask.unsqueeze(1)
        print(
            f"{i}: {loss.item():.2f} new score {resp_news.mean().item():.2f} "
            f"old score {resp_base.item():.2f} RF img dist{img_dists.mean().item():.2f}")
    return dzs_init, dzs.detach().cpu(), img_dists.detach().cpu(), \
           curimgs.detach().cpu(), resp_news.detach().cpu()


def latent_explore_batch(G, Dist, scorer, z_base, resp_base, img_base, rfmaptsr, opts, outdir, repn=20, ):
    os.makedirs(outdir, exist_ok=True)
    dz_init_col = []
    dz_col = []
    score_col = []
    imdist_col = []
    for i in range(repn):
        dzs_init, dzs, img_dists, curimgs, scores = latent_diversity_explore_wRF_fixval(G,
                            Dist, scorer, z_base, resp_base, img_base, rfmaptsr, **opts)
        save_imgrid(curimgs, join(outdir, f"proto_divers_{i}.png"))
        save_imgrid(curimgs * rfmaptsr.cpu(), join(outdir, f"proto_divers_wRF_{i}.png"))
        dz_init_col.append(dzs_init)
        dz_col.append(dzs)
        score_col.append(scores)
        imdist_col.append(img_dists)

    dz_init_tsr = torch.cat(dz_init_col, dim=0)
    dz_final_tsr = torch.cat(dz_col, dim=0)
    score_vec = torch.cat(score_col, dim=0)
    imdist_vec = torch.cat(imdist_col, dim=0)
    savedict = {"dz_init": dz_init_tsr, "dz_final": dz_final_tsr, "score": score_vec, "imdist": imdist_vec,
                "z_base": z_base, "score_base": resp_base, "rfmaptsr": rfmaptsr, "opts": opts, }
    torch.save(savedict, join(outdir, "diversity_dz_score.pt"))
    return EasyDict(savedict)


def search_peak_evol(G, scorer, nstep=100):
    """Initial evolution CMA to find the peak"""
    resp_all = []
    z_all = []
    optimizer = CholeskyCMAES(4096, population_size=None, init_sigma=3.0)
    z_arr = np.zeros((1, 4096))  # optimizer.init_x
    pbar = tqdm(range(nstep))
    for i in pbar:
        imgs = G.visualize(torch.tensor(z_arr).float().cuda())
        resp = scorer.score(imgs, )
        z_arr_new = optimizer.step_simple(resp, z_arr)
        z_arr = z_arr_new
        resp_all.append(resp)
        z_all.append(z_arr)
        print(f"{i}: {resp.mean():.2f}+-{resp.std():.2f}")

    resp_all = np.concatenate(resp_all, axis=0)
    z_all = np.concatenate(z_all, axis=0)
    z_base = torch.tensor(z_all.mean(axis=0, keepdims=True)).float().cuda()
    img_base = G.visualize(z_base)
    resp_base = scorer.score(img_base, )
    return z_base, img_base, resp_base, resp_all, z_all


def search_peak_gradient(G, scorer, z_base, resp_base, nstep=200):
    """Initial gradient based evolution Adam to find the peak"""
    dz = 0.1 * torch.randn(1, 4096).cuda()
    dz.requires_grad_()
    optimizer = Adam([dz], lr=0.1)
    for i in tqdm(range(nstep)):
        optimizer.zero_grad()
        curimg = G.visualize(z_base + dz)
        resp_new = scorer.score_tsr_wgrad(curimg, )
        # img_dist = Dist(img_base, curimg)
        loss = - resp_new
        loss.backward()
        optimizer.step()
        print(f"{i}: {loss.item():.2f} new score {resp_new.item():.2f} "
              f"old score {resp_base.item():.2f}")
    # show_imgrid(curimgs)
    z_base = z_base + dz.detach().clone()
    z_base.detach_()
    img_base = G.visualize(z_base)
    resp_base = scorer.score(img_base, )
    return z_base, img_base, resp_base


def calc_rfmap(scorer, rf_dir, label=None, use_fit=True, device="cuda",):
    if label is None:
        label = "%s-%d"%(scorer.layer.replace(".Bottleneck", "-Btn").strip("."), scorer.chan)
    gradAmpmap = grad_RF_estimate(scorer.model, scorer.layer, (slice(None), scorer.unit_x, scorer.unit_y),
                                  input_size=scorer.inputsize, device=device, show=False, reps=200, batch=4)
    show_gradmap(gradAmpmap, )
    fitdict = fit_2dgauss(gradAmpmap, label, outdir=rf_dir, plot=True)
    rfmap = fitdict.fitmap if use_fit else fitdict.gradAmpmap
    rfmap /= rfmap.max()
    rfmaptsr = torch.from_numpy(rfmap).float().cuda().unsqueeze(0).unsqueeze(0)
    rfmaptsr = F.interpolate(rfmaptsr, (256, 256), mode="bilinear", align_corners=True)
    rfmap_full = rfmaptsr.cpu()[0, 0, :].unsqueeze(2).numpy()
    return rfmaptsr, rfmap_full, fitdict
