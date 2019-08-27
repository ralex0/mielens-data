import os
import json
import pickle
import warnings
from collections import OrderedDict

import numpy as np

import holopy as hp
from holopy.core.io import load_average
from holopy.core.process import subimage, normalize, center_find

from lmfit.minimizer import MinimizerResult, Parameters

RGB_CHANNEL = None
HOLOGRAM_SIZE = 256
HERE = os.path.dirname(__file__)


def load_mcmc_result_PS_mieonly(frame=1, fmt='pkl'):
    folder = 'fits/Polystyrene2-4um-60xWater-042919/'
    file = ('polystyrene-mieonly-frame={}-size=256-npx=10000-mcmc'.format(frame))
    return _load_mcmc_result(folder + file, fmt)


def load_mcmc_result_PS_mielens(frame=1, fmt='pkl'):
    folder = 'fits/Polystyrene2-4um-60xWater-042919/'
    file = ('polystyrene-mielensalpha-frame={}-size=256-npx=10000-mcmc'.format(frame))
    return _load_mcmc_result(folder + file, fmt)


def _load_mcmc_result(file, fmt):
    if fmt == 'json':
        return load_MinimizerResult_from_json(file + '.json')
    elif fmt == 'pkl':
        return load_pickle(file + '.pkl')


def load_pickle(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)


def save_pickle(obj, filename):
    with open(filename, 'wb') as f:
        return pickle.dump(obj, f)


def load_json(filename):
    with open(filename, 'r') as f:
        return json.load(f, object_pairs_hook=OrderedDict)


def save_json(obj, filename):
    if isinstance(obj, MinimizerResult):
        _save_MinimizerResult_to_json(obj, filename)
    else:
        with open(filename, 'w') as f:
            json.dump(obj, f, indent=4)


def load_polystyrene_sedimentation_data(size=HOLOGRAM_SIZE, holonums=range(1000),
                                        recenter=True):
    camera_resolution = 5.6983 # px / um
    metadata = {'spacing' : 1 / camera_resolution,
                'medium_index' : 1.33,
                'illum_wavelen' : .660,
                'illum_polarization' : (1, 0)}
    position = [270, 370]  #  leaves all the particles mostly in the hologram
    paths = ["data/Polystyrene2-4um-60xWater-042919/raw/im"
             +  zfill(num, 4) + ".tif" for num in holonums]
    refimg = hp.load_image(paths[0], **metadata)
    bkg = load_bkg(
        "data/Polystyrene2-4um-60xWater-042919/raw/",
        bg_prefix='bg', refimg=refimg)
    dark = load_dark(
        "data/Polystyrene2-4um-60xWater-042919/raw/",
        df_prefix='dark', refimg=refimg)
    holos = []
    for path in paths:
        this_holo = load_bgdivide_crop(
            path=path, metadata=metadata, particle_position=position,
            bkg=bkg, dark=dark, size=size, recenter=recenter)[0]
        holos.append(this_holo)
    return holos


def fastload_polystyrene_sedimentation_data(size=HOLOGRAM_SIZE, recenter=True):
    camera_resolution = 5.6983 # px / um
    metadata = {'spacing' : 1 / camera_resolution,
                'medium_index' : 1.33,
                'illum_wavelen' : .660,
                'illum_polarization' : (1, 0)}
    folder = 'data/Polystyrene2-4um-60xWater-042919/processed-{}'.format(size)
    if recenter is False: folder += '-uncentered'
    folder = os.path.join(HERE, folder)
    paths = [os.path.join(folder + '/im' + zfill(num) + '.tif')
             for num in range(1000)]
    data = [hp.load_image(path, **metadata) for path in paths]
    return data


def load_polystyrene_sedimentation_params():
    mo_fits = load_json('fits/Polystyrene2-4um-60xWater-042919/mieonly_fits.json')
    ml_fits = load_json('fits/Polystyrene2-4um-60xWater-042919/mielensalpha_fits.json')
    return mo_fits, ml_fits


def load_silica_sedimentation_params():
    mo_fits = load_json('fits/Silica1um-60xWater-080619/mieonly_fits.json')
    ml_fits = load_json('fits/Silica1um-60xWater-080619/mielensalpha_fits.json')
    return mo_fits, ml_fits


def load_polystyrene_sedimentation_fits(date_subdir="04-02"):
    mo_fits = load_pickle('fits/sedimentation/newdata/fits_mo4.pkl')
    ml_fits = load_pickle('fits/sedimentation/newdata/fits_ml3.pkl')
    return mo_fits, ml_fits


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
        size=HOLOGRAM_SIZE, recenter=True):
    data = hp.load_image(path, channel=channel, **metadata)
    data = bg_correct(data, bkg, dark)

    if recenter:
        bbox = subimage(data, particle_position, size)
        bbox_corner = np.array([bbox.x.min(), bbox.y.min()])
        found_position = np.round(
            center_find(bbox) + bbox_corner / metadata['spacing'])
        data = subimage(data, found_position, size)
    else:
        data = subimage(data, particle_position, size)
    data = normalize(data)
    if recenter:
        return data, found_position
    return data, None

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
    serialized_result = load_json(filename)
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


def save_fits_to_json(fits, filename):
    params = OrderedDict()
    for i, fit in enumerate(fits):
        params.update({str(i): fit.params.valuesdict()})
    save_json(params, filename)


def load_example_data():
    imagepath = hp.core.io.get_example_data_path('image01.jpg')
    raw_holo = hp.load_image(imagepath, spacing = 0.0851, medium_index = 1.33,
                             illum_wavelen = 0.66, illum_polarization = (1,0))
    bgpath = hp.core.io.get_example_data_path(['bg01.jpg', 'bg02.jpg', 'bg03.jpg'])
    bg = hp.core.io.load_average(bgpath, refimg = raw_holo)
    holo = hp.core.process.bg_correct(raw_holo, bg)
    holo = hp.core.process.subimage(holo, [250,250], 200)
    holo = hp.core.process.normalize(holo)
    return holo


def load_gold_example_data():
    return normalize(hp.load(hp.core.io.get_example_data_path('image0001.h5')))


def load_silica_sedimentation_data(size=HOLOGRAM_SIZE, holonums=range(1000),
                                        recenter=True):
    camera_resolution = 5.6983 * 1.5 # px / um
    metadata = {'spacing' : 1 / camera_resolution,
                'medium_index' : 1.33,
                'illum_wavelen' : .660,
                'illum_polarization' : (1, 0)}
    position = [650, 587]  # leaves the particle in the hologram for most frames
    paths = ["data/Silica1um-60xWater-080619/raw0[x1.5]/im"
             +  zfill(num, 4) + ".tif" for num in holonums]
    refimg = hp.load_image(paths[0], **metadata)
    bkg = load_bkg(
        "data/Silica1um-60xWater-080619/raw0[x1.5]/",
        bg_prefix='bg', refimg=refimg)
    dark = load_dark(
        "data/Silica1um-60xWater-080619/raw0[x1.5]/",
        df_prefix='dark', refimg=refimg)
    holos = []
    all_positions = []
    new_pos = None
    for path in paths:
        this_holo, new_pos = load_bgdivide_crop(
            path=path, metadata=metadata, particle_position=position,
            bkg=bkg, dark=dark, size=size, recenter=recenter)
        holos.append(this_holo)
        if new_pos is not None: position = new_pos; all_positions.append([tuple(new_pos)])
    return holos

def centerfind_xy_positions_silica(size=HOLOGRAM_SIZE, holonums=range(1000)):
    camera_resolution = 5.6983 * 1.5 # px / um
    metadata = {'spacing' : 1 / camera_resolution,
                'medium_index' : 1.33,
                'illum_wavelen' : .660,
                'illum_polarization' : (1, 0)}
    position = [650, 587] # leaves the particle in the hologram for most frames
    paths = ["data/Silica1um-60xWater-080619/raw0[x1.5]/im"
             +  zfill(num, 4) + ".tif" for num in holonums]
    refimg = hp.load_image(paths[0], **metadata)
    bkg = load_bkg(
        "data/Silica1um-60xWater-080619/raw0[x1.5]/",
        bg_prefix='bg', refimg=refimg)
    dark = load_dark(
        "data/Silica1um-60xWater-080619/raw0[x1.5]/",
        df_prefix='dark', refimg=refimg)
    all_positions = []
    for path in paths:
        this_holo, position = load_bgdivide_crop(
            path=path, metadata=metadata, particle_position=position,
            bkg=bkg, dark=dark, size=size, recenter=True)
        all_positions.append(tuple(position))
    return all_positions

def fastload_silica_sedimentation_data(size=HOLOGRAM_SIZE, recenter=True):
    camera_resolution = 5.6983# * 1.5 # px / um
    metadata = {'spacing' : 1 / camera_resolution,
                'medium_index' : 1.33,
                'illum_wavelen' : .660,
                'illum_polarization' : (1, 0)}
    folder = 'data/Silica1um-60xWater-021519/processed-{}'.format(size)
    if recenter is False: folder += '-uncentered'
    folder = os.path.join(HERE, folder)
    paths = [os.path.join(folder + '/im' + zfill(num) + '.tif')
             for num in range(100)]
    data = [hp.load_image(path, **metadata) for path in paths]
    return data
