"""Code for analyze and visualize experiment data.

"""
import pandas as pd
import torch
import os
import re
from os.path import join
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from easydict import EasyDict
from lpips import LPIPS
from scipy.stats import pearsonr, spearmanr
# from core.utils.GAN_utils import upconvGAN
from core.utils.plot_utils import show_imgrid, save_imgrid, save_imgrid_by_row


def sweep_folder(root, dirnm_pattern=".*_max_abinit$", sum_sfx="summary"):
    """ Post hoc sorting of the results.
        1. Create summary folder with `sum_sfx` suffix.
        2. Sweep through subfolders with `dirnm_pattern` regex
        3. Collect data from "diversity_dz_score.pt" into `sumdict`
            * 'imdist', 'score', 'z' are stored as a list. 
            * "z_base", "score_base", "rfmaptsr" are stored as a single entry. 
            * Format the list as a batch first torch tensor. 
        4. Save `sumdict` in `sumdir`.
    """
    sumdir = join(root, sum_sfx)
    if os.path.exists(join(sumdir, "diversity_z_summary.pt")):
        print("Summary already exists. Skipping.")
        sumdict = torch.load(join(sumdir, "diversity_z_summary.pt"))
        return sumdict, sumdir

    os.makedirs(sumdir, exist_ok=True)
    repatt = re.compile(dirnm_pattern)  # (".*_max$") (".*_min$")
    dirnms = os.listdir(root)
    sumdict = EasyDict({'imdist': [], 'score': [], 'z': []})
    for dirnm in dirnms:
        if not re.match(repatt, dirnm):
            continue
        saveD = EasyDict(torch.load(join(root, dirnm, "diversity_dz_score.pt")))
        z_final = saveD.dz_final + saveD.z_base.cpu()  # apply perturbation to `z_base`
        sumdict['z'].append(z_final)
        for k in ['imdist', 'score']:
            sumdict[k].append(saveD[k])
        for k in ["z_base", "score_base", "rfmaptsr"]:
            sumdict[k] = saveD[k].cpu()

    # format list into torch tensor. 
    for k in sumdict:
        if isinstance(sumdict[k], list):
            if len(sumdict[k]) > 0:
                sumdict[k] = torch.cat(sumdict[k], dim=0)
            else:
                sumdict[k] = torch.empty(0)

    torch.save(sumdict, join(sumdir, "diversity_z_summary.pt"))
    return sumdict, sumdir


def visualize_proto_by_level(G, sumdict, sumdir, bin_width=0.10, relwidth=0.25,
                             sampimgN=6, show=False):
    """ Summarize the images per level in column and concat them in a matrix
    columns are activation levels, rows are different samples from a level.
    Images are showed under RF masks.
    """
    rfmaptsr = sumdict.rfmaptsr.cuda()
    proto_all_tsr = []
    for bin_c in np.arange(0.0, 1.10, 0.10):
        bin_r = bin_c + relwidth * bin_width
        bin_l = bin_c - relwidth * bin_width
        idx_mask = (sumdict.score >= bin_l * sumdict.score_base) * \
              (sumdict.score < bin_r * sumdict.score_base)
        # idx = idx_mask.nonzero().squeeze()
        z_bin = sumdict.z[idx_mask][:sampimgN]
        imgtsrs_rf = (G.visualize(z_bin.cuda()) * rfmaptsr.cuda()).cpu()
        if z_bin.shape[0] < sampimgN:
            padimgN = sampimgN - imgtsrs_rf.shape[0]
            imgtsrs_rf = torch.cat((imgtsrs_rf,
                    torch.zeros(padimgN, *imgtsrs_rf.shape[1:])), dim=0)
        if show:
            show_imgrid(imgtsrs_rf, nrow=1,)
        save_imgrid(imgtsrs_rf, join(sumdir, "proto_in_range_%0.2f-%.2f.png" % (bin_l, bin_r)),
                    nrow=1,)
        proto_all_tsr.append(imgtsrs_rf)

    save_imgrid(torch.cat(proto_all_tsr, dim=0),
                join(sumdir, "proto_level_progression.png", ),
                nrow=sampimgN, rowfirst=False)
    return proto_all_tsr


def visualize_score_imdist(sumdict, sumdir, ):
    fig, ax = plt.subplots(1, 1, figsize=(6, 5))
    rval, pval = spearmanr(sumdict.score, sumdict.imdist)
    rval_pos, pval_pos = spearmanr(sumdict.score[sumdict.score > 0.0],
                                   sumdict.imdist[sumdict.score > 0.0])
    ax.plot(sumdict.score / sumdict.score_base, sumdict.imdist, '.', alpha=0.5)
    plt.xlabel("score / max score")
    plt.ylabel("LPIPS imdist")
    plt.title("Image distance to prototype vs score\n"
                 "Spearman r=%.3f p=%.1e\n"
                 "Spearman (excld 0) r=%.3f p=%.1e" % (rval, pval, rval_pos, pval_pos))
    plt.savefig(join(sumdir, "imdist_vs_score.png"))
    plt.show()
    # Score histogram
    fig, ax = plt.subplots(1, 1, figsize=(6, 5))
    plt.hist(sumdict.score.numpy(), bins=40)
    plt.vlines(sumdict.score_base.item(), *plt.ylim(), color="red", linestyles="dashed")
    plt.xlabel("unit score")
    plt.ylabel("count")
    plt.title("unit score marginal distribution")
    plt.savefig(join(sumdir, "unit_score_dist_max_abinit.png"))
    plt.show()


def calc_proto_diversity_per_bin(G, Dist, sumdict, sumdir, bin_width=0.10,
                                 distsampleN=40, lpips_batch=64):
    """ Bin the results and summarize the prototypes within the bin.
    Compute the LPIPS and pixel level image diversity of images in the bin
    """
    rfmaptsr = sumdict.rfmaptsr.cuda()
    pixdist_dict = {}
    lpipsdist_dict = {}
    lpipsdistmat_dict = {}
    df = pd.DataFrame()
    for bin_c in np.arange(0.0, 1.10, 0.10):
        # left side and right side of bin.
        bin_r = bin_c + 0.4 * bin_width
        bin_l = bin_c - 0.4 * bin_width
        idx_mask = (sumdict.score >= bin_l * sumdict.score_base) * \
              (sumdict.score < bin_r * sumdict.score_base)

        # idx = idx_mask.nonzero().squeeze()
        imdist_bin = sumdict.imdist[idx_mask]
        score_bin = sumdict.score[idx_mask]
        z_bin = sumdict.z[idx_mask]
        print("%0.2f-%0.2f: %d imdist %.3f+-%.3f  score %.3f+-%.3f" % (bin_l, bin_r, idx_mask.sum(),
              imdist_bin.mean().item(), imdist_bin.std().item(),
              score_bin.mean().item(), score_bin.std().item()))

        if z_bin.shape[0] > 1:  # cannot compute pairwise distance with one sample;
            imgtsrs = G.visualize_batch(z_bin[:distsampleN, :].cuda())
            imgtsrs_rf = imgtsrs * rfmaptsr.cpu()
            pixdist = torch.pdist(imgtsrs_rf.flatten(start_dim=1))
            print("pairwise pixel dist %.3f+-%.3f N=%d" % (pixdist.mean(), pixdist.std(), len(pixdist)))
            pixdist_dict[bin_c] = pixdist
            # DONE: this is slow. Can we do it in batch?
            # calculate lpips distance matrix row by row.
            # distmat_bin = []
            # for i in range(imgtsrs.shape[0]):
            #     dist_in_bin = Dist(imgtsrs.cuda() * rfmaptsr, imgtsrs[i:i + 1].cuda() * rfmaptsr).cpu().squeeze()
            #     distmat_bin.append(dist_in_bin)
            #
            # distmat_bin = torch.stack(distmat_bin, dim=0)
            # Batch processing version, much faster!
            distmat_bin = Dist.forward_distmat(imgtsrs_rf.cuda(), None, batch_size=lpips_batch).cpu().squeeze()
            mask = torch.triu(torch.ones(*distmat_bin.shape, dtype=torch.bool), diagonal=1, )
            pairwise_dist = distmat_bin[mask]
            print("pairwise dist %.3f+-%.3f N=%d" % (pairwise_dist.mean(), pairwise_dist.std(), len(pairwise_dist)))
            lpipsdist_dict[bin_c] = pairwise_dist
            lpipsdistmat_dict[bin_c] = distmat_bin
            df_part = pd.DataFrame({"bin_c": bin_c, "bin_l": bin_l,"bin_r": bin_r,
                          "pixdist": pixdist, "lpipsdist": pairwise_dist})
            df = df_part if df.empty else pd.concat((df, df_part), axis=0)
        else:
            print("cannot compute pairwise distance with one or zero sample;")

    torch.save({"pixdist": pixdist_dict, "lpipsdist": lpipsdist_dict, "lpipsdistmat": lpipsdistmat_dict},
               join(sumdir, "imgdist_by_bin_dict.pt"))
    df.to_csv(join(sumdir, "imgdist_by_bin.csv"))
    df["bin_label"] = df.bin_c.apply(lambda c: "%0.1f" % c)
    return df, pixdist_dict, lpipsdist_dict, lpipsdistmat_dict


def visualize_diversity_by_bin(df, sumdir):
    """ Plot the lpips and pixel distance diversity as a function of activation level.
    df: Take in output from `calc_proto_diversity_per_bin`
    """
    rval, pval = spearmanr(df.bin_c, df.lpipsdist)
    rval_pos, pval_pos = spearmanr(df.bin_c[df.bin_c > 0.0], df.lpipsdist[df.bin_c > 0.0])
    figh, ax = plt.subplots(1, 1, figsize=(6, 5))
    sns.violinplot(x="bin_label", y="lpipsdist", data=df)
    ax.set_xlabel("activation level (bin center)")
    ax.set_ylabel("lpips dist among prototypes")
    ax.set_title("LPIPS diversity ~ activation level\n"
                 "Spearman r=%.3f p=%.1e\n"
                 "Spearman (excld 0) r=%.3f p=%.1e" % (rval, pval, rval_pos, pval_pos))
    plt.savefig(join(sumdir, "pairwise_lpips_dist_by_bin.png"))
    plt.show()
    rval, pval = spearmanr(df.bin_c, df.pixdist)
    rval_pos, pval_pos = spearmanr(df.bin_c[df.bin_c > 0.0], df.pixdist[df.bin_c > 0.0])
    figh, ax = plt.subplots(1, 1, figsize=(6, 5))
    sns.violinplot(x="bin_label", y="pixdist", data=df)
    ax.set_xlabel("activation level (bin center)")
    ax.set_ylabel("pixel dist among prototypes")
    ax.set_title("pixel diversity ~ activation level\n"
                 "Spearman r=%.3f p=%.1e\n"
                 "Spearman (excld 0) r=%.3f p=%.1e" % (rval, pval, rval_pos, pval_pos))
    plt.savefig(join(sumdir, "pairwise_pixel_dist_by_bin.png"))
    plt.show()


def filter_visualize_codes(G, outdir, thresh=2.5, err=None, subdir="sorted", abinit=True):
    """Filter out codes with certain score and only present these codees in the subdir with name `sorted` """
    S = EasyDict(torch.load(join(outdir, "diversity_dz_score.pt")))
    imdist_good = S.imdist[S.score > thresh]
    imdist_bad = S.imdist[S.score < thresh]
    score_good = S.score[S.score > thresh]
    score_bad = S.score[S.score < thresh]

    print(f"imdist good {imdist_good.mean():.2f}+-{imdist_good.std():.2f}\t"
          f"{imdist_bad.mean():.2f}+-{imdist_bad.std():.2f}")
    print(f"score good {score_good.mean():.2f}+-{score_good.std():.2f}\t"
          f"{score_bad.mean():.2f}+-{score_bad.std():.2f}")
    rho, pval = pearsonr(S.score, S.imdist)
    print(f"Corr between score and im dist to proto {rho:.3f} P={pval:.3f} (All samples)")

    os.makedirs(join(outdir, subdir), exist_ok=True)
    sortidx = torch.argsort(- S.score)
    score_sort = S.score[sortidx]
    msk = score_sort > thresh
    if err is not None:
        msk = torch.abs(score_sort - thresh) < err
    score_sort = score_sort[msk]
    imdist_sort = S.imdist[sortidx][msk]
    dz_final_sort = S.dz_final[sortidx, :][msk, :]
    zs = dz_final_sort if abinit else dz_final_sort + S.z_base.cpu()
    if len(zs) == 0:
        print("no valid code! under the error tolerance. ")
        return S
    imgs = G.visualize_batch(zs)
    save_imgrid(imgs, join(outdir, subdir, "proto_divers.png"))
    save_imgrid(imgs * S.rfmaptsr.cpu(), join(outdir, subdir, "proto_divers_wRF.png"))
    save_imgrid_by_row(imgs * S.rfmaptsr.cpu(), join(outdir, subdir, "proto_divers_wRF.png"), n_row=5)
    S_new = S
    S_new.score = score_sort
    S_new.imdist = imdist_sort
    S_new.dz_final = dz_final_sort
    S_new.dz_init = S.dz_init[sortidx, :][msk, :]
    torch.save(S_new, join(outdir, subdir, "diversity_dz_score.pt"))
    df = pd.DataFrame({"score": S_new.score, "imdist2proto": S_new.imdist})
    df.to_csv(join(outdir, subdir, "score_dist_summary.csv"))
    if len(S_new.score) > 1 and len(S_new.imdist) > 1:
        rho, pval = pearsonr(S_new.score, S_new.imdist)
        print(f"Corr between score and im dist to proto {rho:.3f} P={pval:.3f} (After filter)")
    else:
        print("not enough entry to compute correlation")
    # torch.save(dict(score=score_sort, imdist=imdist_sort, dz_final=dz_final_sort,
    #                 dz_init=S.dz_init[sortidx, :][msk, :], z_base=S.z_base,
    #                 score_base=S.score_base, rfmaptsr=S.rfmaptsr, opts=S.opts),
    #            join(outdir, "sorted", "diversity_dz_score.pt"))
    return S_new

def torch_cosine_mat(X, Y=None):
    if Y is None:
        Y = X
    X = X.view(X.shape[0], -1)
    Y = Y.view(Y.shape[0], -1)
    return (X @ Y.T) / torch.norm(X, dim=1, keepdim=True) / torch.norm(Y, dim=1, keepdim=True).T



#%% Rename folder structure.
def _rename_folder_structure(root):
    # root = r"E:\insilico_exps\proto_diversity\resnet50_linf8\layer3-Btn5-5_rf"
    dirnms = os.listdir(root)
    for dirnm in dirnms:
        if os.path.isdir(join(root, dirnm)) and dirnm.startswith("layer3-Btn5-5_rf_"):
            os.rename(join(root, dirnm), join(root, dirnm.replace("layer3-Btn5-5_rf_", "")))