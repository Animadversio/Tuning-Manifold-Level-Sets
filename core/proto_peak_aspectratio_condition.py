from os.path import join

from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import pickle as pkl
from easydict import EasyDict as edict
from core.utils.plot_utils import saveallforms
from core.proto_analysis_lib import sweep_folder
from scipy.stats import pearsonr, spearmanr
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
layerlist= [".layer4.Bottleneck2",
            ".layer4.Bottleneck0",
            ".layer3.Bottleneck5",
            ".layer3.Bottleneck2",
            ".layer2.Bottleneck3",
            ".layer1.Bottleneck2",]

netname = "resnet50_linf8"
outroot = join(r"E:\insilico_exps\proto_diversity", netname)
syndir = join(outroot, "synopsis")
cachedir = join(outroot, "cache")
sumfigdir = join(outroot, "summary")

#%%

def digitize_score_dist(scorearr, imdistarr, sfx, score_base, bin_width=0.10, ):
    """ bin the score and imdist into bins of width bin_width (fraction of `score_base`)

    """
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
    """ create the dataframe of `level by imdist stats` for each bin
    imdist in differernt optimization condtions are listed in columns
    """
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


def corr_reg_analysis_sumdicts(sumdicts):
    df_col = {}
    score_base = sumdicts.max.score_base.item()
    for sfx in ("min", "none", "max", "max_abinit", "none_abinit"):
        sumdict = sumdicts[sfx]
        if len(sumdict.score) == 0 and len(sumdict.imdist) == 0:
            # empty dict if that condition is not run.
            S = dict(corr=np.nan, corr_excl0=np.nan,
                     slope=np.nan, slope_excl0=np.nan,
                     bias=np.nan, bias_excl0=np.nan, )
        else:
            excl0msk = sumdict.score > 0.0
            rval, _ = spearmanr(sumdict.score, sumdict.imdist)
            rval_excl0, _ = spearmanr(sumdict.score[excl0msk], sumdict.imdist[excl0msk])
            slope, bias = np.polyfit(sumdict.score / score_base, sumdict.imdist, 1)
            slope_excl0, bias_excl0 = np.polyfit(sumdict.score[excl0msk] / score_base, sumdict.imdist[excl0msk], 1)
            S = dict(corr=rval, corr_excl0=rval_excl0,
                slope=slope, slope_excl0=slope_excl0,
                bias=bias, bias_excl0=bias_excl0,)
        df_col[sfx] = S
    return pd.DataFrame(df_col)
#%% Export figures for layers
#%%
chan_list = [*range(10, 20)] # *range(10)  *range(10), *range(20, 60),
for layer in layerlist:
    for chan in tqdm(chan_list):
        unit_tup = (netname, layer, chan)
        unitlabel = "%s-%d" % (layer.replace(".Bottleneck", "-Btn").strip("."), chan)
        outrf_dir = join(outroot, unitlabel+"_rf")
        sumdicts = edict()
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
        corr_reg_df = corr_reg_analysis_sumdicts(sumdicts)
        corr_reg_df.to_csv(join(syndir, "%s_imdist_score_corr_reg.csv" % unitlabel))
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

#%%
def _regress2level(df, varname, ):
    """wrapper to deal with nan value in the dataframe """
    valmsk = (~np.isnan(df[varname])) & (df["level_fr"] > 0.0)
    if valmsk.sum() > 0:
        slp, bias = np.polyfit(df["level_fr"][valmsk], df[varname][valmsk], 1)
        return slp, bias
    else:
        return np.nan, np.nan
#%% Export figures for layers
# Collect stats for each unit
netname = "resnet50_linf8"
outroot = join(r"E:\insilico_exps\proto_diversity", netname)
chan_list = [*range(60)] #[*range(10), *range(20,60)] #*range(10, 20)
syn_col = []
for layeri, layer in zip([6, 5, 4, 3, 2, 1], layerlist, ):
    for chan in tqdm(chan_list):
        unit_tup = (netname, layer, chan)
        unitlabel = "%s-%d" % (layer.replace(".Bottleneck", "-Btn").strip("."), chan)
        outrf_dir = join(outroot, unitlabel+"_rf")
        df = pd.read_csv(join(syndir, "%s_imdist_score_curve.csv" % unitlabel))
        df_corr_reg = pd.read_csv(join(syndir, "%s_imdist_score_corr_reg.csv" % unitlabel), index_col=0)
        df_corr_reg = df_corr_reg.T
        # raise Exception("stop")
        #%% filter out entries with too few samples
        df.loc[df.N_max < 10, "imdist_m_max"] = np.nan
        df.loc[df.N_min < 10, "imdist_m_min"] = np.nan
        df.loc[df.N_none < 10, "imdist_m_none"] = np.nan
        df.loc[df.N_max_abinit < 10, "imdist_m_max_abinit"] = np.nan
        df.loc[df.N_none_abinit < 10, "imdist_m_none_abinit"] = np.nan
        #%%
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
                  maxmin_ratio_m=maxmin_rat[-5:-1].mean(), maxmin_ratio_peak=maxmin_rat[-2:].mean(),
                  maxnone_ratio_m=maxnone_rat[-5:-1].mean(), maxnone_ratio_peak=maxnone_rat[-2:].mean(),
                  nonemin_ratio_m=nonemin_rat[-5:-1].mean(), nonemin_ratio_peak=nonemin_rat[-2:].mean(),
                  glbloc_max_ratio_m=glbloc_max_rat[-5:-1].mean(), glbloc_max_ratio_peak=glbloc_max_rat[-2:].mean(),
                  glbloc_none_ratio_m=glbloc_none_rat[-5:-1].mean(), glbloc_none_ratio_peak=glbloc_none_rat[-2:].mean(),
                  slp_min=slp_min, bias_min=bias_min, slp_max=slp_max, bias_max=bias_max, slp_mab=slp_mab, bias_mab=bias_mab,
                  slp_maxmin_ratio=slp_maxmin_ratio, bias_glbloc_ratio=bias_glbloc_ratio,
                  corr_excl0_maxabinit=df_corr_reg.loc["max_abinit", "corr_excl0"], corr_excl0_max=df_corr_reg.loc["max", "corr_excl0"], corr_excl0_min=df_corr_reg.loc["min", "corr_excl0"],
                  slp_excl0_maxabinit=df_corr_reg.loc["max_abinit", "slope_excl0"], slp_excl0_max=df_corr_reg.loc["max", "slope_excl0"], slp_excl0_min=df_corr_reg.loc["min", "slope_excl0"],
                  bias_excl0_maxabinit=df_corr_reg.loc["max_abinit", "bias_excl0"], bias_excl0_max=df_corr_reg.loc["max", "bias_excl0"], bias_excl0_min=df_corr_reg.loc["min", "bias_excl0"],
                    )
        syn_col.append(S)
df_syn = pd.DataFrame(syn_col)
#%%
df_syn["layer_short"] = df_syn.layer.apply(lambda x: x.replace(".Bottleneck", "-Btn").strip("."))
df_syn.to_csv(join(syndir, "stats_synopsis_robust.csv"))
df_summary = df_syn.groupby("layer").agg(["mean", "sem"]).T
df_summary.to_csv(join(syndir, "stats_summary_per_area_robust.csv"))
# df.to_csv(join(syndir, "%s_imdist_score_curve.csv"%unitlabel))
#%%
def plot_var_summary(df_syn, varnm, by="layer_short", ax=None,
                     max_cutoff_q=0.995, savefig=True, **kwargs, ):
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(4, 6))
    ax = sns.stripplot(x=by, y=varnm, data=df_syn, jitter=0.2, ax=ax, alpha=0.6, **kwargs,)
    plt.xticks(rotation=35)
    plt.ylim(bottom=None, top=df_syn[varnm].quantile(max_cutoff_q))
    plt.tight_layout()
    if savefig:
        saveallforms(sumfigdir, f"{varnm}_strip_layer_robust")
    plt.show()
    return ax.figure, ax
#%%
plot_var_summary(df_syn, "maxmin_ratio_m")
#%%
plot_var_summary(df_syn, "maxmin_ratio_m")
plot_var_summary(df_syn, "maxnone_ratio_m")
plot_var_summary(df_syn, "nonemin_ratio_m")
#%%
plot_var_summary(df_syn, "glbloc_none_ratio_m")
#%%
figh, ax = plot_var_summary(df_syn, "maxmin_ratio_m", savefig=False, palette="flare")
figh.set_size_inches(3.2, 6)
ax.axhline(1, color="k", linestyle="--")
ax.set_ylim(bottom=0, )
saveallforms(sumfigdir, "maxmin_ratio_m_strip_layer_robust_edit", figh=figh)
figh.show()
#%%
figh, ax = plot_var_summary(df_syn, "glbloc_max_ratio_m", savefig=False, palette="flare")
figh.set_size_inches(3.2, 6)
ax.axhline(1, color="k", linestyle="--")
ax.set_ylim(bottom=0, )
saveallforms(sumfigdir, "glbloc_max_ratio_m_strip_layer_robust_edit", figh=figh)
figh.show()
#%%

fig, ax = plot_var_summary(df_syn, "corr_excl0_max", savefig=False)
fig, ax = plot_var_summary(df_syn, "corr_excl0_maxabinit", ax=ax)
# plot_var_summary(df_syn, "corr_excl0_max")
# plot_var_summary(df_syn, "slp_excl0_maxabinit")
#%%
# plot_var_summary(df_syn, "slp_mab")
plot_var_summary(df_syn, "bias_mab")

#%%
print(df_syn.groupby("layer").agg({"maxmin_ratio_m": ["mean", "std","count"]}))
#%%
print(df_syn.groupby("layer").agg({"glbloc_max_ratio_m": ["mean", "std","count"]}))
#%% Filter invalid or unstable data.


