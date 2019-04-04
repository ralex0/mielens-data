import pickle
import json
import os
import warnings

import numpy as np

import holopy as hp
from holopy.core.io import load_average
from holopy.core.process import subimage, normalize

from lmfit.minimizer import MinimizerResult, Parameters

RGB_CHANNEL = 'all'
HOLOGRAM_SIZE = 100

def load_mcmc_result_PS_mieonly():
    return load_pickle('fits/sedimentation/mcmc/ps_mcmc_mo_1.pkl')


def load_mcmc_result_PS_mielens():
    return load_pickle('fits/sedimentation/mcmc/ps_mcmc_ml_1.pkl')


def load_mcmc_result_Si_mieonly():
    return load_pickle('fits/sedimentation/mcmc/si_mcmc_mo_29.pkl')


def load_mcmc_result_Si_mielens():
    return load_pickle('fits/sedimentation/mcmc/si_mcmc_ml_29.pkl')


def load_PTmcmc_result_PS_mieonly():
    """PS: 0001.tiff 
    {'burn': 0, 'steps': 1000, 'nwalkers': 100, 'thin': 1, 'workers': 4, 'ntemps': 5, 'npixels': 100}
    """
    return load_pickle('fits/sedimentation/mcmc/PT-4-4-19/ps_mcmc_mo_pt.pkl')


def load_PTmcmc_result_PS_mielens():
    """PS: 0001.tiff 
    {'burn': 0, 'steps': 1000, 'nwalkers': 100, 'thin': 1, 'workers': 4, 'ntemps': 5, 'npixels': 100}
    """
    return load_pickle('fits/sedimentation/mcmc/PT-4-4-19/ps_mcmc_ml_pt.pkl')


def load_PTmcmc_result_Si_mieonly():
    """Si: 0029.tiff
    {'burn': 0, 'steps': 1000, 'nwalkers': 100, 'thin': 1, 'workers': 4, 'ntemps': 3, 'npixels': 100}
    """
    return load_pickle('fits/sedimentation/mcmc/PT-4-4-19/si_mcmc_mo_pt.pkl')


def load_PTmcmc_result_Si_mielens():
    """Si: 0029.tiff
    {'burn': 0, 'steps': 1000, 'nwalkers': 100, 'thin': 1, 'workers': 4, 'ntemps': 3, 'npixels': 100}
    """
    return load_pickle('fits/sedimentation/mcmc/PT-4-4-19/si_mcmc_ml_pt.pkl')


def load_silica_sedimentation_params(date_subdir="04-02"):
    mo_fits = load_json(f"fits/sedimentation/{date_subdir}/mieonly_sedimentation_params_Si.json")
    ml_fits = load_json(f"fits/sedimentation/{date_subdir}/mielens_sedimentation_params_Si.json")
    return mo_fits, ml_fits
    

def load_polystyrene_sedimentation_params(date_subdir="04-02"):
    mo_fits = load_json(f"fits/sedimentation/{date_subdir}/mieonly_sedimentation_params_PS.json")
    ml_fits = load_json(f"fits/sedimentation/{date_subdir}/mielens_sedimentation_params_PS.json")
    return mo_fits, ml_fits


def load_silica_sedimentation_fits(date_subdir="04-02"):
    mo_fits = [load_pickle(f"fits/sedimentation/{date_subdir}/Si_mieonly/{zfill(num, 4)}.pkl") for num in range(43)]
    ml_fits = [load_pickle(f"fits/sedimentation/{date_subdir}/Si_mielens/{zfill(num, 4)}.pkl") for num in range(100)]
    return mo_fits, ml_fits
    

def load_polystyrene_sedimentation_fits(date_subdir="04-02"):
    mo_fits = [load_pickle(f"fits/sedimentation/{date_subdir}/PS_mieonly/{zfill(num, 4)}.pkl") for num in range(20)]
    ml_fits = [load_pickle(f"fits/sedimentation/{date_subdir}/PS_mielens/{zfill(num, 4)}.pkl") for num in range(50)]
    return mo_fits, ml_fits


def load_pickle(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)

def save_pickle(obj, filename):
    with open(filename, 'wb') as f:
        return pickle.dump(obj, f)

def load_json(filename):
    with open(filename, 'r') as f:
        return json.load(f)

def save_json(obj, filename):
    if isinstance(obj, MinimizerResult):
        _save_MinimizerResult_to_json(obj, filename)
    else:
        with open(filename, 'w') as f:
            json.dump(obj, f)

def load_silica_sedimentation_data():
    camera_resolution = 5.6983 # px / um
    metadata = {'spacing' : 1 / camera_resolution,
                    'medium_index' : 1.33,
                    'illum_wavelen' : .660,
                    'illum_polarization' : (1, 0)}
    position = [553, 725]

    holonums = range(100)
    zpos = np.linspace(38, -12, 100)
    paths = ["data/Silica1um-60xWater-021519/raw/image"
             +  zfill(num, 4) + ".tif" for num in holonums]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        refimg = hp.load_image(paths[0], **metadata)
        bkg = load_bkg(
            "data/Silica1um-60xWater-021519/raw/bg/",
            bg_prefix='bg', refimg=refimg)  # 10 s! all holopy
        dark = load_dark(
            "data/Silica1um-60xWater-021519/raw/dark/",
            df_prefix='dark', refimg=refimg)  # 8.7 s! all holopy
        holos = [load_bgdivide_crop(path=path, metadata=metadata,
                                    particle_position=position,
                                    bkg=bkg, dark=dark, size=140)
                 for path in paths]
    return holos, zpos


def load_polystyrene_sedimentation_data():
    camera_resolution = 5.6983 # px / um
    metadata = {'spacing' : 1 / camera_resolution,
                    'medium_index' : 1.33,
                    'illum_wavelen' : .660,
                    'illum_polarization' : (1, 0)}
    position = [242, 266]

    holonums = range(50)
    zpos = np.linspace(20, -12, 50)
    paths = ["data/Polystyrene2-4um-60xWater-012419/raw/image"
             +  zfill(num, 4) + ".tif" for num in holonums]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        refimg = hp.load_image(paths[0], **metadata)
        bkg = load_bkg(
            "data/Polystyrene2-4um-60xWater-012419/raw/",
            bg_prefix='bg', refimg=refimg)
        dark = load_dark(
            "data/Polystyrene2-4um-60xWater-012419/raw/",
            df_prefix='dark', refimg=refimg)
        holos = [load_bgdivide_crop(path=path, metadata=metadata,
                                    particle_position=position,
                                    bkg=bkg, dark=dark, size=100)
                 for path in paths]
    return holos, zpos


def load_bkg(path, bg_prefix, refimg):
    bkg_paths = get_bkg_paths(path, bg_prefix)
    bkg = load_average(bkg_paths, refimg=refimg, channel=RGB_CHANNEL)
    return bkg


def load_dark(path, df_prefix, refimg):
    return load_bkg(path, df_prefix, refimg) if df_prefix is not None else None


def get_bkg_paths(path, bg_prefix):
    subdir = os.path.dirname(path)
    bkg_paths = [subdir + '/' + pth for pth in os.listdir(subdir) if bg_prefix in pth]
    return bkg_paths


def load_bgdivide_crop(
        path, metadata, particle_position, bkg, dark, channel=RGB_CHANNEL,
        size=HOLOGRAM_SIZE):
    data = hp.load_image(path, channel=channel, **metadata)
    data = bg_correct(data, bkg, dark)
    data = subimage(data, particle_position[::-1], size)
    data = normalize(data)
    return data

def bg_correct(raw, bg, df=None):
    if df is None:
        df = raw.copy()
        df[:] = 0
    denominator = bg - df
    denominator.values = np.clip(denominator.values, 1e-7, np.inf)
    holo = (raw - df) / denominator
    holo = hp.core.copy_metadata(raw, holo)
    return holo

def zfill(n, nzeros=4):
    return str(n).rjust(nzeros, '0')


RESULT_ATTRS = ['params', 'status', 'var_names', 'covar', 'init_vals', 
                'init_values', 'nfev', 'success', 'errorbars', 'message', 'ier',
                'lmdif_message', 'nvarys', 'ndata', 'nfree', 'residual',
                'chisqr', 'redchi', 'aic', 'bic', 'chain', 'method', 'lnprob']


def _save_MinimizerResult_to_json(result, filename):
    serialized_result = _serialize_MinimizerResult(result)
    with open(filename, 'w') as f:
        json.dump(serialized_result, f, indent=4)


def _serialize_MinimizerResult(result):
    attrs = [a for a in RESULT_ATTRS if hasattr(result, a)]
    serialized = {a: getattr(result, a) for a in attrs}
    serialized['params'] = serialized['params'].dumps()

    for k, v in serialized.items():
        if isinstance(v, np.ndarray):
            serialized[k] = v.tolist()
        elif isinstance(v, bool) or isinstance(v, np.bool_):
            serialized[k] = 'True' if serialized[k] else 'False'
    return serialized


def load_MinimizerResult_from_json(filename):
    with open(filename, 'r') as f:
        serialized_result = json.load(f)
    unserialized_result = _unserialize_MinimizerResult(serialized_result)
    return MinimizerResult(**unserialized_result)


def _unserialize_MinimizerResult(serialized):
    unserialized = serialized
    unserialized['params'] = Parameters().loads(serialized['params'])
    for k in ['covar', 'residual', 'init_vals', 'lnprob', 'chain']:
        if k in serialized:
            unserialized[k] = np.array(serialized[k])
    if 'success' in serialized:        
        unserialized['success'] = True if serialized['success'] == 'True' else False
    if 'errorbars' in serialized:
        unserialized['errorbars'] = (np.bool_(True) 
                                     if serialized['errorbars'] == 'True'
                                     else np.bool_(False))
    return unserialized