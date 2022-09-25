"""
Summarize the topological signatures of the level sets for in silico manifold experiments

"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from glob import glob
from os.path import join
from easydict import EasyDict as edict
from invivo_analysis import *

rootdir = r"E:\Cluster_Backup\manif_allchan"
# catalogue of all the experiments.
df_all = pd.read_csv(join(rootdir, "summary", "resnet50_linf_8_ManifExpNonParamSum_RFfit.csv"))
#%%
def _load_proto_info(tabrow, layerdir=None, layerfulldir=None):
    """
    Example:
        proto_dir = r"E:\Cluster_Backup\manif_allchan\prototypes"
        layerdir = join(proto_dir, f"vgg16_{layer}_manifold-")
        layerfulldir = join(r"E:\Cluster_Backup\manif_allchan", f"vgg16_{layer}_manifold-")
        protoimg, Edata, Mdata = _load_proto_info(unitrow, layerdir, layerfulldir)

    :param tabrow:
    :param layerdir:
    :param layerfulldir:
    :return:
        prototype image (np),
        Evol data (easydict),
        Manif data (numpy array 4x21x21)
    """
    if isinstance(tabrow, pd.Series):
        layer_long, unitid = tabrow.layer, tabrow.iCh
    elif isinstance(tabrow, pd.DataFrame):
        layer_long, unitid = tabrow.layer.iloc[0], tabrow.iCh[0]
    else:
        raise ValueError("tab must be a pandas.DataFrame or pandas.Series")
    if layerdir is None:
        layerdir = join(proto_dir, f"resnet50_linf_8_{layer_long}_manifold-")
    if layerfulldir is None:
        layerfulldir = join(r"E:\Cluster_Backup\manif_allchan", f"resnet50_linf_8_{layer_long}_manifold-")
    if "Linearfc" in layer_long or ".layer4.Bottleneck" in layer_long:
        suffix = "original"
    else:
        suffix = "rf_fit"
    if "resnet50_linf_8" in layerdir:
        layer = layer_long
    filenametemplate = glob(join(layerdir, f"*_{suffix}.png"))[0]
    unitpos = filenametemplate.split("\\")[-1].split("_")[3:5]
    unit = unitid
    if "fc" in layer:
        img = plt.imread(join(layerdir, f"proto_{layer}_{unit:d}_{suffix}.png"))
        Edata = np.load(join(layerfulldir, f"Manifold_set_{layer}_{unit:d}_{suffix}.npz"))
        Mdata = np.load(join(layerfulldir, f"Manifold_score_{layer}_{unit:d}_{suffix}.npy"))
    else:
        img = plt.imread(join(layerdir, f"proto_{layer}_{unit:d}_{unitpos[0]}_{unitpos[1]}_{suffix if suffix=='original' else suffix+'_full'}.png"))
        Edata = np.load(join(layerfulldir, f"Manifold_set_{layer}_{unit:d}_{unitpos[0]}_{unitpos[1]}_{suffix}.npz"))
        Mdata = np.load(join(layerfulldir, f"Manifold_score_{layer}_{unit:d}_{unitpos[0]}_{unitpos[1]}_{suffix}.npy"))
    return img, edict(Edata), Mdata
# from core.insilico_EM_data_utils import _load_proto_montage, _load_proto_info
tabrow = df_all.iloc[18000]
layer = tabrow.layer
proto_dir = r"E:\Cluster_Backup\manif_allchan\prototypes"
img, Edata_row, Mdata_row = _load_proto_info(tabrow, )
#%%
actmap = Mdata_row[0]
df, lvlset_dict = analyze_levelsets_topology(actmap, nlevels=21, )
plot_levelsets_topology(df, explabel=tabrow.explabel)
plt.show()
#%%
# plot_spherical_levelset(lvlset_dict, )
visualize_levelsets_all(actmap, )
plt.show()
#%%
from tqdm import tqdm
syn_col = []
for i, tabrow in tqdm(df_all.iterrows()):
    img, Edata_row, Mdata_row = _load_proto_info(tabrow, )
    actmap = Mdata_row[0]
    df, lvlset_dict = analyze_levelsets_topology(actmap, nlevels=21, )
    syn_col.append(pd.concat([df.n_loop.rename('loop_{}'.format),
                              df.n_line.rename('line_{}'.format)], ).astype("int32").T)
    # plot_levelsets_topology(df, explabel=tabrow.explabel)
    # plt.show()
    # if i == 100:
    #     break
#%%
syndf = pd.DataFrame(syn_col, )
#%%
# syndf.astype("int32")

syndf.to_csv(join(rootdir, "summary", "resnet50_linf_8_Manif_levelset_topology.csv"), )
#%%
outdir = r"E:\OneDrive - Harvard University\Manifold_insilico_topology\summary"
syndf.to_csv(join(outdir, "resnet50_linf_8_Manif_levelset_topology.csv"), )
#%%
syndf_all = pd.concat([df_all.layer, df_all.iCh, syndf], axis=1) # df_all.actmax, df_all.actmin,
#%%
layer_topo_mean = syndf_all.drop("iCh", axis=1).groupby("layer").mean()
layer_topo_std  = syndf_all.drop("iCh", axis=1).groupby("layer").std()
layer_topo_sem  = syndf_all.drop("iCh", axis=1).groupby("layer").sem()
#%%
syndf_all_branch = pd.DataFrame(syndf.iloc[:, :21].values +
                                 syndf.iloc[:, 21:].values).rename('branch_{}'.format, axis=1)
syndf_all_branch = pd.concat([df_all.layer, df_all.iCh, syndf_all_branch], axis=1)
branch_mean = syndf_all_branch.drop("iCh", axis=1).groupby("layer").mean()
branch_std  = syndf_all_branch.drop("iCh", axis=1).groupby("layer").std()
branch_sem  = syndf_all_branch.drop("iCh", axis=1).groupby("layer").sem()
#%%
layer_topo_mean = pd.concat([layer_topo_mean, branch_mean], axis=1)
layer_topo_std  = pd.concat([layer_topo_std, branch_std], axis=1)
layer_topo_sem  = pd.concat([layer_topo_sem, branch_sem], axis=1)
#%%
def shaded_errorbar(ax, x, y, yerr, label=None, **kwargs):
    ax.plot(x, y, label=label, **kwargs)
    ax.fill_between(x, y - yerr, y + yerr, alpha=0.2, label=None, **kwargs)
#%%
from core.utils import saveallforms
from matplotlib.ticker import MaxNLocator
for layer in layer_topo_mean.index:
    figh, axh = plt.subplots(figsize=(5, 4))
    plt.errorbar(range(21), layer_topo_mean.loc[layer, ][:21],
                 yerr=layer_topo_std.loc[layer,][:21], fmt='o-', label="N loops", alpha=0.7)
    plt.errorbar(range(21), layer_topo_mean.loc[layer, ][21:],
                 yerr=layer_topo_std.loc[layer,][21:], fmt='o-', label="N lines", alpha=0.7)
    # plt.errorbar(range(21), branch_mean.loc[layer, ],
    #              yerr=branch_std.loc[layer, ], fmt='o-', label="N branches", alpha=0.7)
    # set xticks as integers
    axh.xaxis.set_major_locator(MaxNLocator(integer=True))
    axh.yaxis.set_major_locator(MaxNLocator(integer=True))
    plt.legend()
    plt.ylim(bottom=0)
    plt.xlabel("Level set index")
    plt.ylabel("Number of segments")
    plt.title("Mean Topological Signature\n"+layer)
    saveallforms(outdir, f"resnet50_linf8_mean_{layer}_topo_pure", figh)
    plt.show()
#%%
# get color sequence for each layer, 10
from matplotlib import cm
# cm.get_cmap('tab10')
def short_layername(layer):
    return layer[1:].replace("Bottleneck", "B").replace("Linear", "").replace("ReLU", "")


def plot_mean_topology_across_layer(layer_topo_mean, layer_topo_sem, layer_order, err="sem",
                                    cmap=None, outdir=outdir):
    if cmap is None:
        cmap = cm.get_cmap("jet")
    colors = [cmap((i / 9)) for i in range(10)]
    for idxslc, name_topo in zip([slice(0, 21), slice(21, 42), slice(42, 63)],
                                 ["n_loop", "n_line", "n_branch"],):
        figh, axh = plt.subplots(figsize=(6, 4))
        for li, layer in enumerate(layer_order):
            shaded_errorbar(axh, range(21), layer_topo_mean.loc[layer, ][idxslc],
                            layer_topo_sem.loc[layer, ][idxslc],
                            label=short_layername(layer), color=colors[li])

        axh.xaxis.set_major_locator(MaxNLocator(integer=True))
        # axh.yaxis.set_major_locator(MaxNLocator(integer=True))
        plt.legend()
        plt.ylim(bottom=0)
        plt.xlabel("Level set index")
        plt.ylabel("Number of segments")
        plt.title(f"Mean {name_topo} across layers")
        saveallforms(outdir, f"resnet50_linf8_topo_mean_{err}_{name_topo}", figh)
        plt.show()
#%%
layer_order = ['.ReLUrelu', '.layer1.Bottleneck1', '.layer2.Bottleneck0',
       '.layer2.Bottleneck2', '.layer3.Bottleneck0', '.layer3.Bottleneck2',
       '.layer3.Bottleneck4', '.layer4.Bottleneck0', '.layer4.Bottleneck2', '.Linearfc', ]
#%%
plot_mean_topology_across_layer(layer_topo_mean, layer_topo_sem, layer_order, err="sem")
plot_mean_topology_across_layer(layer_topo_mean, layer_topo_std, layer_order, err="std")
