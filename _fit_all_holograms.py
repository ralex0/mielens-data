import os

import numpy as np

import matplotlib.pyplot as plt

import holopy as hp
from holopy.scattering import calc_holo, Sphere
from holopy.scattering.theory import MieLens
from holopy.core.io import load_average
from holopy.core.process import bg_correct, subimage

import sys
sys.path.append('../mielens-data')

import _mielensfit as mlf

CAMERA_RESOLUTION = 5.9633 # pixels per unit length
DEFAULT_METADATA = {'spacing' : 1 / CAMERA_RESOLUTION,
                    'medium_index' : 1.34,
                    'illum_wavelen' : .532,
                    'illum_polarization' : (1, 0)}
RGB_CHANNEL = 1
HOLOGRAM_SIZE = 100

### Load/Correct/Crop all data
def load_bgdivide_crop(path, metadata, particle_position, bg_prefix="bg", channel=RGB_CHANNEL):
    data = hp.load_image(path, channel=channel, **metadata)
    bkg = load_bkg(path, bg_prefix, refimg=data)
    data = bg_correct(data, bkg)
    data = subimage(data, particle_position[::-1], HOLOGRAM_SIZE)
    return data

def load_bgdivide_crop_all(subdir, metadata, particle_position, 
                           data_prefix="holo", bg_prefix="bg"):
    paths = get_img_paths(subdir, data_prefix)
    data = [load_bgdivide_crop(path, metadata, particle_position, bg_prefix) 
            for path in paths]
    return data

def load_bkg(path, bg_prefix, refimg):
    bkg_paths = get_bkg_paths(path, bg_prefix)
    bkg = load_average(bkg_paths, refimg=refimg, channel=RGB_CHANNEL, noise_sd=1.0)
    return bkg 

def get_bkg_paths(path, bg_prefix):
    subdir = os.path.dirname(path)
    bkg_paths = [subdir + '/' + pth for pth in os.listdir(subdir) if bg_prefix in pth]
    return bkg_paths

def get_img_paths(subdir, image_tag):
    imgpaths = [subdir + '/' + path for path in os.listdir("./{}".format(subdir)) 
                if image_tag in path]
    return imgpaths

### Load test data
def _load_test_holo_above():
    Si_metadata = DEFAULT_METADATA
    Si_location = [158, 610] 
    return load_bgdivide_crop(path="11-9/Silica/holo_23.png", 
                              metadata=Si_metadata, 
                              particle_position=Si_location, 
                              bg_prefix="bg_bottom")

def _load_test_holo_below():
    Si_metadata = DEFAULT_METADATA
    Si_location = [158, 610] 
    return load_bgdivide_crop(path="11-9/Silica/holo_87.png", 
                              metadata=Si_metadata, 
                              particle_position=Si_location, 
                              bg_prefix="bg_bottom")


def _load_reza_PS():
    resolution = 1/.194 # pixels per um
    metadata = {'spacing' : 1 / resolution,
                'medium_index' : 1.33,
                'illum_wavelen' : .660,
                'illum_polarization' : (1, 0)}
    PS_location = [255, 255]

    data1 = load_bgdivide_crop(path="rezaPS/image0050.tif", metadata=metadata, 
                               particle_position=PS_location, bg_prefix="bg")
    data2 = load_bgdivide_crop(path="rezaPS/image0060.tif", metadata=metadata, 
                               particle_position=PS_location, bg_prefix="bg")
    data3 = load_bgdivide_crop(path="rezaPS/image0070.tif", metadata=metadata, 
                               particle_position=PS_location, bg_prefix="bg")
    return data1, data2, data3


if __name__ == '__main__':
    Si_zpos = np.array([-30.075,
               -29.500, -29.000, -28.500, -28.000, -27.500, -27.000, -26.475, -26.000, -25.500, -25.000,
               -24.425, -24.000, -23.500, -23.000, -22.500, -22.000, -21.500, -21.000, -20.500, -20.000,
               -19.500, -19.000, -18.500, -18.000, -17.525, -17.000, -16.500, -16.000, -15.525, -15.000,
               -14.500, -14.000, -13.500, -13.000, -12.500, -12.000, -11.500, -11.000, -10.475, -10.000,
               -9.500, -9.000, -8.500, -8.000, -7.475, -7.025, -6.500, -6.000, -5.500, -5.000,
               -4.500, -4.000, -3.500, -3.025, -2.525, -2.000, -1.475, -1.000, -0.500, 0.000,
               0.500, 1.000, 1.500, 2.000, 2.500, 3.000, 3.500, 4.000, 4.500, 5.000,
               5.500, 6.025, 6.525, 7.000, 7.500, 8.025, 8.500, 9.000, 9.500, 10.000,
               10.500, 11.000, 11.500, 12.000, 12.500, 13.000, 13.500, 14.000, 14.500, 15.000,
               15.500, 16.000, 16.500, 17.000, 17.500, 18.000, 18.500, 19.000, 19.500, 20.000,
               20.500, 21.000, 21.500, 22.025, 22.500, 23.000, 23.500, 24.000, 24.500, 25.000,
               25.500, 26.000, 26.500, 27.000, 27.500, 28.000, 28.525, 29.000, 29.500, 30.000,
               30.500, 31.000, 31.500, 32.025, 32.500, 33.000, 33.500, 34.000, 34.500, 35.000,
               35.500, 36.000, 36.500, 37.000])

    #data_1 = _load_test_holo_above()
    #data_2 = _load_test_holo_below()

    reza1, reza2, reza3 = _load_reza_PS()

    guesses1 = {'x': 9.713038139387498,
                'y': 10.456691922232963,
                'z': 4.618498169589564 * .9,
                'n': 1.5734090037527397 * .9,
                'r': 0.5405409574507445* .9,
                'lens_angle': 0.5570046628134516,
                'image_norm': 0.9955129503439557,
                'illum_wavelength': 0.66}


    guesses2 = {'x': 9.723597975477213,
                'y': 10.572893476526572,
                'z': 10.592281131261418 * .9,
                'n': 1.563604754489514 * .9,
                'r': 0.5672417122947906 * .9,
                'lens_angle': 0.5295070067298675,
                'image_norm': 1.0214227644944853,
                'illum_wavelength': 0.66}


    guesses3 = {'x': 9.778775888957426,
                'y': 10.69820362679027,
                'z': 16.73559163136148 * .9,
                'n': 1.555005032286686 * .9,
                'r': 0.5737841199543253 * .9,
                'lens_angle': 0.531536983740261,
                'image_norm': 1.0101610472232756,
                'illum_wavelength': 0.66}

    fit1_ml = mlf._fit_with_mielens_scipy(reza1, guesses1)
    fit2_ml = mlf._fit_with_mielens_scipy(reza2, guesses2)
    fit3_ml = mlf._fit_with_mielens_scipy(reza3, guesses3)

    fit1_mo = mlf._fit_with_mieonly_nmp(reza1, guesses1)
    fit2_mo = mlf._fit_with_mieonly_nmp(reza2, guesses2)
    fit3_mo = mlf._fit_with_mieonly_nmp(reza3, guesses3)
    
    holo_dt1 = reza1.values.squeeze()
    holo_dt2 = reza2.values.squeeze()
    holo_dt3 = reza3.values.squeeze()

    holo_fit1_ml = calc_holo(reza1, fit1_ml.scatterer, theory=MieLens(lens_angle=fit1_ml.intervals[-1].guess)).values.squeeze()
    holo_fit2_ml = calc_holo(reza2, fit2_ml.scatterer, theory=MieLens(lens_angle=fit2_ml.intervals[-1].guess)).values.squeeze()
    holo_fit3_ml = calc_holo(reza3, fit3_ml.scatterer, theory=MieLens(lens_angle=fit3_ml.intervals[-1].guess)).values.squeeze()

    holo_fit1_mo = calc_holo(reza1, fit1_mo.scatterer, scaling=fit1_mo.intervals[-1].guess).values.squeeze()
    holo_fit2_mo = calc_holo(reza2, fit2_mo.scatterer, scaling=fit2_mo.intervals[-1].guess).values.squeeze()
    holo_fit3_mo = calc_holo(reza3, fit3_mo.scatterer, scaling=fit3_mo.intervals[-1].guess).values.squeeze()

    res1_ml = mlf._calc_sum_res_sq_mielens(reza1, fit1_ml) 
    res2_ml = mlf._calc_sum_res_sq_mielens(reza2, fit2_ml) 
    res3_ml = mlf._calc_sum_res_sq_mielens(reza3, fit3_ml) 

    res1_mo = mlf._calc_sum_res_sq_mieonly(reza1, fit1_mo) 
    res2_mo = mlf._calc_sum_res_sq_mieonly(reza2, fit2_mo) 
    res3_mo = mlf._calc_sum_res_sq_mieonly(reza3, fit3_mo) 

    titles1 = ["image0050", "Best Fit mielens, err^2 = {:.3f}".format(res1_ml), "Best Fit mieonly, err^2 = {:.3f}".format(res1_mo)]
    titles2 = ["image0060", "Best Fit mielens, err^2 = {:.3f}".format(res2_ml), "Best Fit mieonly, err^2 = {:.3f}".format(res2_mo)]
    titles3 = ["image0070", "Best Fit mielens, err^2 = {:.3f}".format(res3_ml), "Best Fit mieonly, err^2 = {:.3f}".format(res3_mo)]

    mlf.plot_three_things(holo_dt1, holo_fit1_ml, holo_fit1_mo, titles1)
    mlf.plot_three_things(holo_dt2, holo_fit2_ml, holo_fit2_mo, titles2)
    mlf.plot_three_things(holo_dt3, holo_fit3_ml, holo_fit3_mo, titles3)

