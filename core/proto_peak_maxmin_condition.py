from os.path import join

import matplotlib.pyplot as plt
from core.utils.plot_utils import saveallforms
from core.proto_analysis_lib import sweep_folder
from easydict import EasyDict

syndir = r"E:\insilico_exps\proto_diversity\resnet50_linf8\synopsis"

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

for unit_tup in unitlist:
    netname = unit_tup[0]
    layer = unit_tup[1]
    chan  = unit_tup[2]
    unitlabel = "%s-%d" % (layer.replace(".Bottleneck", "-Btn").strip("."), chan)
    outroot = join(r"E:\insilico_exps\proto_diversity", netname)
    outrf_dir = join(outroot, unitlabel+"_rf")
    sumdicts = EasyDict()
    for suffix in ["min", "max", "max_abinit"]:
        sumdicts[suffix], sumdir = sweep_folder(outrf_dir, dirnm_pattern=f"fix.*_{suffix}$",
                                       sum_sfx=f"summary_{suffix}")
        # visualize_proto_by_level(G, sumdict, sumdir, bin_width=0.10, relwidth=0.25, )
        # visualize_score_imdist(sumdict, sumdir, )
        # df, pixdist_dict, lpipsdist_dict, lpipsdistmat_dict = calc_proto_diversity_per_bin(G, Dist, sumdict, sumdir,
        #                                                            bin_width=0.10, distsampleN=40)
        # visualize_diversity_by_bin(df, sumdir)
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
    plt.show()
