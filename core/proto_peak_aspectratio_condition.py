from os.path import join

import matplotlib.pyplot as plt
import numpy as np

from core.utils.plot_utils import saveallforms
from core.proto_analysis_lib import sweep_folder
from easydict import EasyDict
#%%

# unit_tup = ("resnet50_linf8", ".layer4.Bottleneck2", 5, 3, 3)
unitlist = [("resnet50_linf8", ".layer4.Bottleneck2", 10, 3, 3),
            ("resnet50_linf8", ".layer4.Bottleneck1", 5, 3, 3),
            ("resnet50_linf8", ".layer3.Bottleneck5", 5, 7, 7),
            ("resnet50_linf8", ".layer3.Bottleneck2", 5, 7, 7),
            ("resnet50_linf8", ".layer2.Bottleneck3", 5, 13, 13),
            ("resnet50_linf8", ".layer3.Bottleneck0", 5, 7, 7),
            ("resnet50_linf8", ".layer3.Bottleneck5", 10, 7, 7),
            ("resnet50_linf8", ".layer4.Bottleneck0", 10, 3, 3),
            ("resnet50_linf8", ".layer2.Bottleneck1", 5, 13, 13),
            ("resnet50_linf8", ".layer2.Bottleneck1", 10, 13, 13), # some problem in min
            ("resnet50_linf8", ".layer2.Bottleneck3", 10, 13, 13),
            ("resnet50_linf8", ".layer3.Bottleneck0", 10, 7, 7),
            ("resnet50_linf8", ".layer3.Bottleneck2", 10, 7, 7),
            ("resnet50_linf8", ".layer3.Bottleneck5", 10, 7, 7),
            ("resnet50_linf8", ".layer3.Bottleneck5", 15, 7, 7),
            ("resnet50_linf8", ".layer4.Bottleneck0", 15, 3, 3),
            ("resnet50_linf8", ".layer4.Bottleneck2", 5, 3, 3),
            # ("resnet50_linf8", ".layer4.Bottleneck2", 10, 3, 3),
            ]

#%%
#%%
import numpy as np
import pandas as pd
import pickle as pkl
from easydict import EasyDict as edict

def digitize_score_dist(scorearr, imdistarr, sfx, score_base, bin_width=0.10, ):
    #  bins[i-1] <= x < bins[i]
    bini = np.digitize(scorearr / score_base, bins=bin_width*np.arange(0, 12)-0.05)
    S_col = []
    for i in range(1, 12):
        leveli = (i - 1) * bin_width
        # level_act = leveli * score_base
        msk = (bini == i)
        imdist_m = imdistarr[msk].mean().item()
        imdist_s = imdistarr[msk].std().item()
        S = edict()
        S["level"] = leveli
        S["imdist_m_"+sfx] = imdist_m
        S["imdist_s_"+sfx] = imdist_s
        S["N_"+sfx] = imdistarr[msk].shape[0]
        S_col.append(S)
    df = pd.DataFrame(S_col)
    return df


def summarize_sumdicts(sumdicts, bin_width=0.10, ):
    df_col = []
    score_base = sumdicts.max.score_base.item()
    for sfx in ("min", "none", "max", "max_abinit", "none_abinit"):
        sumdict = sumdicts[sfx]
        df_cond = digitize_score_dist(sumdict.score, sumdict.imdist, sfx, score_base, bin_width=bin_width)
        df_col.append(df_cond.drop(columns=["level"]))
    df = pd.concat(df_col, axis=1)
    df["level_fr"] = bin_width * np.arange(0, 11)
    df["level"] = df["level_fr"] * score_base
    return df
#%% Export figures for layers
layerlist= [".layer4.Bottleneck2",
            ".layer4.Bottleneck0",
            ".layer3.Bottleneck5",
            ".layer3.Bottleneck2",
            ".layer2.Bottleneck3",
            ".layer1.Bottleneck2",]
syndir = r"E:\insilico_exps\proto_diversity\resnet50_linf8\synopsis"
cachedir = r"E:\insilico_exps\proto_diversity\resnet50_linf8\cache"
# for unit_tup in unitlist:
#     netname = unit_tup[0]
#     layer = unit_tup[1]
#     chan  = unit_tup[2]
netname = "resnet50_linf8"
outroot = join(r"E:\insilico_exps\proto_diversity", netname)
chan_list = [*range(20, 60), ] #*range(10)
for layer in layerlist:
    for chan in chan_list:
        unit_tup = (netname, layer, chan)
        unitlabel = "%s-%d" % (layer.replace(".Bottleneck", "-Btn").strip("."), chan)
        outrf_dir = join(outroot, unitlabel+"_rf")
        sumdicts = EasyDict()
        for suffix in ["min", "none", "max", "max_abinit", "none_abinit"]:
            sumdicts[suffix], sumdir = sweep_folder(outrf_dir, dirnm_pattern=f"fix.*_{suffix}$",
                                           sum_sfx=f"summary_{suffix}")
            # visualize_proto_by_level(G, sumdict, sumdir, bin_width=0.10, relwidth=0.25, )
            # visualize_score_imdist(sumdict, sumdir, )
            # df, pixdist_dict, lpipsdist_dict, lpipsdistmat_dict = calc_proto_diversity_per_bin(G, Dist, sumdict, sumdir,
            #                                                            bin_width=0.10, distsampleN=40)
            # visualize_diversity_by_bin(df, sumdir)
        #%%
        df = summarize_sumdicts(sumdicts, bin_width=0.10)
        df.to_csv(join(syndir, "%s_imdist_score_curve.csv" % unitlabel))
        # pkl.dump(sumdicts, open(join(cachedir, unitlabel+"_sumdicts.pkl"), "wb"))
        #%%
        score_base = sumdicts.max.score_base
        plt.figure()
        plt.scatter(sumdicts.max.score / score_base, sumdicts.max.imdist,
                    alpha=0.5, label="max")
        plt.scatter(sumdicts.min.score / score_base, sumdicts.min.imdist,
                    alpha=0.5, label="min")
        plt.scatter(sumdicts.max_abinit.score / score_base, sumdicts.max_abinit.imdist,
                    alpha=0.5, label="max_abinit")
        plt.ylabel("LPIPS image distance")
        plt.xlabel("activation level (/ max)")
        plt.title(unit_tup)
        plt.legend()
        saveallforms(syndir, "%s_imdist_score_curve"%unitlabel)
        # plt.show()
        plt.close("all")
        # raise Exception("stop")
#%%
def _regress2level(df, varname, ):
    """wrapper to deal with nan value in the dataframe """
    valmsk = (~np.isnan(df[varname])) & (df["level_fr"] > 0.0)
    slp, bias = np.polyfit(df["level_fr"][valmsk], df[varname][valmsk], 1)
    return slp, bias
#%% Export figures for layers

netname = "resnet50_linf8"
outroot = join(r"E:\insilico_exps\proto_diversity", netname)
chan_list = [*range(10), *range(20,60)] #*range(10, 20)
syn_col = []
for layeri, layer in zip([6,5,4,3,2,1], layerlist, ):
    for chan in chan_list:
        unit_tup = (netname, layer, chan)
        unitlabel = "%s-%d" % (layer.replace(".Bottleneck", "-Btn").strip("."), chan)
        outrf_dir = join(outroot, unitlabel+"_rf")
        df = pd.read_csv(join(syndir, "%s_imdist_score_curve.csv" % unitlabel))
        maxmin_rat = df["imdist_m_max"] / df["imdist_m_min"]
        maxnone_rat = df["imdist_m_max"] / df["imdist_m_none"]
        nonemin_rat = df["imdist_m_none"] / df["imdist_m_min"]
        glbloc_max_rat = df["imdist_m_max_abinit"] / df["imdist_m_max"]
        glbloc_none_rat = df["imdist_m_none_abinit"] / df["imdist_m_none"]
        slp_min, bias_min = _regress2level(df, "imdist_m_min")
        slp_max, bias_max = _regress2level(df, "imdist_m_max")
        slp_mab, bias_mab = _regress2level(df, "imdist_m_max_abinit")
        slp_maxmin_ratio = slp_max / slp_min
        bias_glbloc_ratio = bias_mab / bias_max
        S = edict(layeri=layeri, chan=chan, layer=layer, unitlabel=unitlabel, score_base=df.level.max(),
                  maxmin_ratio_m=maxmin_rat[-5:].mean(), maxmin_ratio_peak=maxmin_rat[-2:].mean(),
                  maxnone_ratio_m=maxnone_rat[-5:].mean(), maxnone_ratio_peak=maxnone_rat[-2:].mean(),
                  nonemin_ratio_m=nonemin_rat[-5:].mean(), nonemin_ratio_peak=nonemin_rat[-2:].mean(),
                  glbloc_max_ratio_m=glbloc_max_rat[-5:].mean(), glbloc_max_ratio_peak=glbloc_max_rat[-2:].mean(),
                  glbloc_none_ratio_m=glbloc_none_rat[-5:].mean(), glbloc_none_ratio_peak=glbloc_none_rat[-2:].mean(),
                  slp_min=slp_min, bias_min=bias_min, slp_max=slp_max, bias_max=bias_max, slp_mab=slp_mab, bias_mab=bias_mab,
                  slp_maxmin_ratio=slp_maxmin_ratio, bias_glbloc_ratio=bias_glbloc_ratio,)
        syn_col.append(S)
df_syn = pd.DataFrame(syn_col)
#%%
df_syn["layer_short"] = df_syn.layer.apply(lambda x: x.replace(".Bottleneck", "-Btn").strip("."))
df_syn.to_csv(join(syndir, "stats_synopsis.csv"))
#%%
df_summary = df_syn.groupby("layer").agg(["mean", "sem"]).T
df_summary.to_csv(join(syndir, "stats_summary_per_area.csv"))
# df.to_csv(join(syndir, "%s_imdist_score_curve.csv"%unitlabel))
#%%
import seaborn as sns
sumfigdir = r"E:\insilico_exps\proto_diversity\resnet50_linf8\summary"
def plot_var_summary(df_syn, varnm, by="layer_short", ax=None, cutoff_q=0.995, **kwargs):
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(4, 6))
    sns.stripplot(x=by, y=varnm, data=df_syn, jitter=0.2, ax=ax, alpha=0.6)
    plt.xticks(rotation=35)
    plt.ylim(None, df_syn[varnm].quantile(cutoff_q))
    plt.tight_layout()
    saveallforms(sumfigdir, f"{varnm}_strip_layer")
    plt.show()
    return ax.figure
#%%
plot_var_summary(df_syn, "maxmin_ratio_m")
plot_var_summary(df_syn, "maxmin_ratio_m")
plot_var_summary(df_syn, "maxnone_ratio_m")
plot_var_summary(df_syn, "nonemin_ratio_m")
plot_var_summary(df_syn, "glbloc_none_ratio_m")
#%%
# plot_var_summary(df_syn, "slp_mab")
plot_var_summary(df_syn, "bias_mab")

#%%
df_col = []
for sfx in ["min", "none", "max", "max_abinit", "none_abinit"]:
    df_cond = digitize_score_dist(sumdicts[sfx].score, sumdicts[sfx].imdist, sfx, score_base, bin_width=0.10, )
    df_col.append(df_cond.drop(columns=["level"]))
df = pd.concat(df_col, axis=1)
df["level_fr"] = 0.1*np.arange(0, 11)
df["level"] = df["level_fr"] * score_base.item()

