"""Small lib for interpolation of the activation map of the Manifold experiment."""
from os.path import join
import numpy as np
import pickle as pkl
from easydict import EasyDict as edict
from invivo_analysis.neural_data_lib import extract_meta_data
from invivo_analysis.neural_data_lib import get_Evol_Manif_stats, load_score_mat
from scipy.interpolate import RectSphereBivariateSpline, SmoothSphereBivariateSpline

savedir = r"E:\OneDrive - Harvard University\Manifold_sphere_interp"


def sphere_interp_actmap(actmap, s, ):
    """Simple version of activation map interpolation.
    s: smoothing factor, could be determined by the sum of squared error (of mean) of
        the map. Coule be estimated by the squared SEM of the single trial responses.
    """
    actmap_poleval0 = actmap[:, 0].mean()
    actmap_polevalpi = actmap[:, -1].mean()
    lats_vec = np.linspace(0, 180, 11)[1: -1] / 180 * np.pi
    lons_vec = np.arange(-90, 90.1, 18) / 180 * np.pi
    lut = RectSphereBivariateSpline(lats_vec, lons_vec, actmap[:, 1:-1].T,
                                    pole_values=(actmap_poleval0, actmap_polevalpi),
                                    pole_exact=False, s=s)
    new_lats_vec = np.arange(0, 180.1, 1) / 180 * np.pi
    new_lons_vec = np.arange(-90, 90.1, 1) / 180 * np.pi
    new_lats, new_lons = np.meshgrid(new_lats_vec, new_lons_vec)
    data_interp = lut.ev(new_lats.ravel(),
                         new_lons.ravel()).reshape(new_lats.shape)
    return data_interp


def sphere_interp_Manifold(Animal, Expi, EStats=None, MStats=None, ):
    """Interpolate the activation map of the Manifold experiment. (Animal, Expi)
    Higher level api.
    """
    if EStats is None or MStats is None:
        EStats, MStats = get_Evol_Manif_stats(Animal)
    # Load the response data
    scorecol_M, imgfullpath_vect_M = load_score_mat(EStats, MStats, Expi, "Manif_avg", wdws=[(50, 200)], stimdrive="S")
    scorecol_M_sgtr, _ = load_score_mat(EStats, MStats, Expi, "Manif_sgtr", wdws=[(50, 200)], stimdrive="S")
    bslcol_M_sgtr, _ = load_score_mat(EStats, MStats, Expi, "Manif_sgtr", wdws=[(0, 40)], stimdrive="S")
    # 1st index is PC2 degree, 2nd index is PC3 degree
    # 1st index is Theta, 2nd index is Phi
    # format the data into map
    actmap = scorecol_M.reshape(11, 11)
    actcolmap = np.array(scorecol_M_sgtr, dtype=object).reshape(11, 11)
    imgfp_mat = np.array(imgfullpath_vect_M).reshape((11, 11))
    bslvec = np.concatenate(bslcol_M_sgtr)
    bslmean = bslvec.mean()
    bslstd = bslvec.std()
    #% Estimate variability from the single trial response.
    # square of st error of the mean
    semsqarray = np.array([np.var(trrsp) / len(trrsp) for trrsp in scorecol_M_sgtr]).reshape(11, 11)
    # sum of square of sem across the interpolated points
    semsqsum = semsqarray[:, 1:-1].sum()  # np.sum([np.var(trrsp) for trrsp in scorecol_M_sgtr])
    #%  Norse pole and south pole value
    actmap_poleval0 = np.concatenate(actcolmap[:, 0]).mean()  # actmap[:, 0].mean()
    actmap_polevalpi = np.concatenate(actcolmap[:, -1]).mean()  # actmap[:, -1].mean()
    #%
    lats_vec = np.linspace(0, 180, 11)[1:-1] / 180 * np.pi
    lons_vec = np.arange(-90, 90.1, 18) / 180 * np.pi
    lut = RectSphereBivariateSpline(lats_vec, lons_vec, actmap[:, 1:-1].T,
                                    pole_values=(actmap_poleval0, actmap_polevalpi),
                                    pole_exact=False, s=semsqsum)
    new_lats_vec = np.arange(0, 180.1, 1) / 180 * np.pi
    new_lons_vec = np.arange(-90, 90.1, 1) / 180 * np.pi
    new_lats, new_lons = np.meshgrid(new_lats_vec, new_lons_vec)
    data_interp = lut.ev(new_lats.ravel(),
                         new_lons.ravel()).reshape(new_lats.shape)
    data_interp = np.maximum(data_interp, 0.0)
    return data_interp, lut, actmap, bslmean


def compute_all_interpolation(savedir=savedir):
    for Animal in ["Alfa", "Beto"]:
        EStats, MStats = get_Evol_Manif_stats(Animal)
        for Expi in range(1, len(EStats)+1):
            data_interp, lut, actmap, bslmean = sphere_interp_Manifold(Animal, Expi, EStats, MStats)
            assert data_interp.min() >= 0
            np.savez(join(savedir, f"{Animal}_{Expi}_interp_map.npz"),
                     data_interp=data_interp, actmap=actmap, bslmean=bslmean)
            pkl.dump(lut, open(join(savedir, f"{Animal}_{Expi}_interp_lut.pkl"), "wb"))


def compute_all_meta(savedir=savedir):
    meta_col = edict()
    for Animal in ["Alfa", "Beto"]:
        EStats, MStats = get_Evol_Manif_stats(Animal)
        meta_col[Animal] = []
        for Expi in range(1, len(EStats) + 1):
            meta, expstr = extract_meta_data(Animal, Expi, EStats, MStats)
            print(expstr)
            meta_col[Animal].append(meta)

    pkl.dump(meta_col, open(join(savedir, "Both_metadata.pkl"), "wb"))
    return meta_col


#% Load precomputed interpolation data
def load_meta(Animal, Expi, savedir=savedir):
    """ load the metadata for all experiments and return the request one. """
    meta_col = pkl.load(open(join(savedir, "Both_metadata.pkl"), "rb"))
    return meta_col[Animal][Expi-1]


def load_data_interp(Animal, Expi, savedir=savedir):
    """ data loading routine from precomputed interpolated tuning map."""
    data = np.load(join(savedir, f"{Animal}_{Expi}_interp_map.npz"), allow_pickle=True)
    data_interp = data["data_interp"]
    actmap = data["actmap"]
    bslmean = data["bslmean"]
    lut = np.load(join(savedir, f"{Animal}_{Expi}_interp_lut.pkl"), allow_pickle=True)
    return data_interp, lut, actmap, bslmean

if __name__ == "__main__":
    #%% Mass calculation all stats
    for Animal in ["Alfa", "Beto"]:
        EStats, MStats = get_Evol_Manif_stats(Animal)
        for Expi in range(1, len(EStats)+1):
            data_interp, lut, actmap, bslmean = sphere_interp_Manifold(Animal, Expi, EStats, MStats)
            assert data_interp.min() >= 0
            np.savez(join(savedir, f"{Animal}_{Expi}_interp_map.npz"),
                     data_interp=data_interp, actmap=actmap, bslmean=bslmean)
            pkl.dump(lut, open(join(savedir, f"{Animal}_{Expi}_interp_lut.pkl"), "wb"))
    #%%
    meta_col = edict()
    for Animal in ["Alfa", "Beto"]:
        EStats, MStats = get_Evol_Manif_stats(Animal)
        meta_col[Animal] = []
        for Expi in range(1, len(EStats)+1):
            meta, expstr = extract_meta_data(Animal, Expi, EStats, MStats)
            print(expstr)
            meta_col[Animal].append(meta)

    pkl.dump(meta_col, open(join(savedir, "Both_metadata.pkl"), "wb"))
    #%%


