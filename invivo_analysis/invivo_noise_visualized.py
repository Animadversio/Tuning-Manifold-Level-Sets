from invivo_analysis.Manif_interp_lib import load_data_interp
from invivo_analysis import get_Evol_Manif_stats, load_score_mat, load_meta
#%%
# for Animal in ["Alfa", "Beto"]:
Animal = "Alfa"
Expi = 3
EStats, MStats = get_Evol_Manif_stats(Animal)
meta = load_meta(Animal, Expi)
data_interp, lut, actmap, bslmean = load_data_interp(Animal, Expi)
#%%
# scorecol_M_sgtr, _ = load_score_mat(EStats, MStats, Expi, "Manif_sgtr", wdws=[(50, 200)], stimdrive="S")
psth_col = MStats[Expi-1].manif.psth.reshape(-1)
act_col = [P[0, 50:200, :].mean(axis=(0)) for P in psth_col]
#%%
import numpy as np
X = np.arange(-90, 90.1, 18)
Y = np.arange(-90, 90.1, 18)
XX, YY = np.meshgrid(X, Y)
Xvec = XX.reshape(-1)
Yvec = YY.reshape(-1)
Xsgtr = [np.ones_like(P) * x for P, x in zip(act_col, Xvec)]
Ysgtr = [np.ones_like(P) * y for P, y in zip(act_col, Yvec)]
Xvec_all = np.concatenate(Xsgtr)
Yvec_all = np.concatenate(Ysgtr)
actvec_all = np.concatenate(act_col)
#%%
import numpy as np
import matplotlib.pyplot as plt
plt.figure()
plt.imshow(actmap)
plt.show()
#%%
from os.path import join
outdir = r"E:\OneDrive - Harvard University\NeurRep2022_NeurIPS\Figures\Manif_invivo_NoiseVisualize"
fig, ax = plt.subplots(subplot_kw={"projection": "3d"}, figsize=[8,8])
# Make data.
X = np.arange(-90, 90.1, 18)
Y = np.arange(-90, 90.1, 18)
X, Y = np.meshgrid(X, Y)
Z = actmap
xfine = np.arange(-90, 90.1)
yfine = np.arange(-90, 90.1)
xxfine, yyfine = np.meshgrid(xfine, yfine)
# Plot the surface.
# surf1 = ax.plot_surface(X, Y, Z, cmap=plt.cm.coolwarm, alpha=0.6,
#                        linewidth=1, antialiased=False)
ax.scatter(Xvec_all, Yvec_all, actvec_all, s=9, c="k", alpha=0.4)
surf2 = ax.plot_surface(xxfine, yyfine, data_interp, cmap=plt.cm.coolwarm, alpha=0.9, linewidth=0, antialiased=False)
ax.set_xlabel("PC2")
ax.set_ylabel("PC3")
ax.set_zlabel("activation (spk/sec)")
ax.view_init(20, 60)
ax.set_title(f"{Animal} Exp{Expi:02d} PrefChan{meta.prefchan}", fontsize=16)
plt.tight_layout()
plt.savefig(join(outdir, f"{Animal}_Exp{Expi:02d}_Manif_noise.png"))
plt.savefig(join(outdir, f"{Animal}_Exp{Expi:02d}_Manif_noise.pdf"))
plt.show()
#%%
Xvec[55:66]
Yvec[55:66]


Xsgtr_part = [np.ones_like(P) * x for P, x in zip(act_col[55:66], Xvec[55:66])]
Xpart_vec = np.concatenate(Xsgtr_part)
actpart_vec = np.concatenate(act_col[55:66])
plt.figure(figsize=[6,6])
plt.scatter(Xpart_vec, actpart_vec, s=9, c="k", alpha=0.7,
            label="Single Trial")
plt.plot(np.arange(-90,90.1,18), actmap[5, :],
         color="blue", marker="o", lw=1.5, label="Mean")
plt.plot(np.arange(-90,90.1), data_interp[90, :],
         color="red", lw=1.5, label="Spherical Interpolated")
plt.axhline(bslmean, linestyle=":", label="Baseline")
plt.xlabel("PC2 (deg)")
plt.ylabel("activation (spk/sec)")
plt.legend()
plt.title(f"{Animal} Exp{Expi:02d} PrefChan{meta.prefchan}")
plt.gca().spines["right"].set_visible(False)
plt.gca().spines["top"].set_visible(False)
plt.tight_layout()

plt.savefig(join(outdir, f"{Animal}_Exp{Expi:02d}_Manif_noise_2d.png"))
plt.savefig(join(outdir, f"{Animal}_Exp{Expi:02d}_Manif_noise_2d.pdf"))
plt.show()