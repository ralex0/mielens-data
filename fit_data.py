import time
import warnings

import numpy as np

import matplotlib.pyplot as plt

import holopy as hp
from holopy.scattering import calc_holo
from holopy.scattering.theory import MieLens

import mielensfit as mlf

def load_few_Si_data():
    camera_resolution = 9.6187 # px / um
    metadata = {'spacing' : 1 / camera_resolution,
                    'medium_index' : 1.34,
                    'illum_wavelen' : .532,
                    'illum_polarization' : (1, 0)}
    position = [839, 382]

    holonums = [28, 38, 48, 78]
    zpos = np.array([15, 10, 5, -10])#np.load('data/Silica-100xOil-120418/zpos.npy')[[holonums]]
    # paths = ["data/Silica-100xOil-120418/raw/im_" +  f"{num}".zfill(3) + ".png"
    paths = ["data/Silica-100xOil-120418/raw/im_" +  "{}".format(str(num)).rjust(3, '0') + ".png"
             for num in holonums]
    holos = [mlf.load_bgdivide_crop(path=path, metadata=metadata,
                                particle_position=position, bg_prefix="bg_bottom")
             for path in paths]
    return holos, zpos

def load_few_PS_data():
    camera_resolution = 9.6187 # px / um
    metadata = {'spacing' : 1 / camera_resolution,
                    'medium_index' : 1.34,
                    'illum_wavelen' : .532,
                    'illum_polarization' : (1, 0)}
    position = [921, 697]

    holonums = [40, 50, 80]
    zpos = np.array([10, 5, -10])#np.load('data/Polystyrene-100xOil-120418/zpos.npy')[[holonums]]
    paths = ["data/Polystyrene-100xOil-120418/greyscale/im" +"{}".format(str(num)).rjust(3, '0') + ".png"
             for num in holonums]
    holos = [mlf.load_bgdivide_crop(path=path, metadata=metadata,
                                particle_position=position, bg_prefix="bg_btm")
             for path in paths]
    return holos, zpos

def load_few_PS_data_old():
    camera_resolution = 5.9633 # px / um
    metadata = {'spacing' : 1 / camera_resolution,
                    'medium_index' : 1.34,
                    'illum_wavelen' : .532,
                    'illum_polarization' : (1, 0)}
    positions = ([515, 497], [520, 497], [523, 497])

    holonums = [60, 80, 100]
    zpos = np.array([15, 10, 5])#np.load('data/Polystyrene-60xWater-111918/zpos.npy')[[holonums]]
    paths = ["data/Polystyrene-60xWater-111918/raw/im" + "{}".format(str(num)).rjust(3, '0') + ".png"
             for num in holonums]
    holos = [mlf.load_bgdivide_crop(path=path, metadata=metadata,
                                particle_position=position, bg_prefix="bg")
             for path, position in zip(paths, positions)]
    return holos, zpos

def load_few_PS_data_reza():
    resolution = 1/.194 # pixels per um
    metadata = {'spacing' : 1 / resolution,
                'medium_index' : 1.33,
                'illum_wavelen' : .660,
                'illum_polarization' : (1, 0)}
    PS_location = [255, 255]

    data1 = mlf.load_bgdivide_crop(path="data/Polystyrene-Reza/image0050.tif", metadata=metadata,
                                   particle_position=PS_location, bg_prefix="bg")
    data2 = mlf.load_bgdivide_crop(path="data/Polystyrene-Reza/image0060.tif", metadata=metadata,
                                   particle_position=PS_location, bg_prefix="bg")
    data3 = mlf.load_bgdivide_crop(path="data/Polystyrene-Reza/image0070.tif", metadata=metadata,
                                   particle_position=PS_location, bg_prefix="bg")

    zpos = np.array([4.618498169589564, 10.592281131261418, 16.73559163136148])
    return [data1, data2, data3], zpos

def load_example_data():
    imagepath = hp.core.io.get_example_data_path('image01.jpg')
    raw_holo = hp.load_image(imagepath, spacing = 0.0851, medium_index = 1.33,
                             illum_wavelen = 0.66, illum_polarization = (1,0))
    bgpath = hp.core.io.get_example_data_path(['bg01.jpg', 'bg02.jpg', 'bg03.jpg'])
    bg = hp.core.io.load_average(bgpath, refimg = raw_holo)
    holo = hp.core.process.bg_correct(raw_holo, bg)
    holo = hp.core.process.subimage(holo, [250,250], 200)
    holo = hp.core.process.normalize(holo)
    zpos = 15.0
    return holo, zpos

def load_PS60xWater_data():
    camera_resolution = 1/.194 # pixels per um
    metadata = {'spacing' : 1 / camera_resolution,
                    'medium_index' : 1.348,
                    'illum_wavelen' : .532,
                    'illum_polarization' : (1, 0)}
    position = [430, 472]

    holonums = range(7)
    zpos = np.array([15, 10, 5, 0, -5, -10, -15])
    paths = ["data/Polystyrene-60xWater-121218/raw/im_" +  "{}".format(str(num)).rjust(3, '0') + ".png"
             for num in holonums]
    holos = [mlf.load_bgdivide_crop(path=path, metadata=metadata,
                                particle_position=position, bg_prefix="bg")
             for path in paths]
    return holos, zpos


def load_Silica60xWater_data():
    camera_resolution = 1/.194 # pixels per um
    metadata = {'spacing' : 1 / camera_resolution,
                    'medium_index' : 1.348,
                    'illum_wavelen' : .532,
                    'illum_polarization' : (1, 0)}
    position = [518, 553]

    holonums = range(7)
    zpos = np.array([15, 10, 5, 0, -5, -10, -15])
    paths = ["data/Silica-60xWater-121218/raw/im_" +  "{}".format(str(num)).rjust(3, '0') + ".png"
             for num in holonums]
    holos = [mlf.load_bgdivide_crop(path=path, metadata=metadata,
                                particle_position=position, bg_prefix="bg")
             for path in paths]
    return holos, zpos

def load_few_PS_data_Jan4():
    camera_resolution = 5.9633 # px / um
    metadata = {'spacing' : 1 / camera_resolution,
                    'medium_index' : 1.348,
                    'illum_wavelen' : .660,
                    'illum_polarization' : (1, 0)}
    position = [294, 395] #z is about  14, 5.5, -12, -20
    #position = [103, 564] # z is about 11, 8.5, 2.5, ....

    holonums = range(4)
    zpos = np.array([14, 5.5, -12, -20])
    paths = ["data/Polystyrene-60xWater-010419/raw0_greyscale/im" + "{}".format(str(num)).rjust(3, '0') + ".png"
             for num in holonums]
    holos = [mlf.load_bgdivide_crop(path=path, metadata=metadata,
                                particle_position=position, bg_prefix="bg", df_prefix="dark")
             for path in paths]
    return holos, zpos


def load_few_PS_data_Jan8():
    camera_resolution = 5.9633 # px / um
    metadata = {'spacing' : 1 / camera_resolution,
                    'medium_index' : 1.348,
                    'illum_wavelen' : .660,
                    'illum_polarization' : (1, 0)}
    position = [617, 291]

    holonums = range(21)
    zpos = np.linspace(-20, 20, 21)
    paths = ["data/Polystyrene-60xWater-010819/raw_greyscale/im_" + "{}".format(str(num)).rjust(3, '0') + ".png"
             for num in holonums]
    holos = [mlf.load_bgdivide_crop(path=path, metadata=metadata,
                                particle_position=position, bg_prefix="bkg", df_prefix="dark")
             for path in paths]
    return holos, zpos

def load_few_PS_data_Jan10():
    camera_resolution = 5.9633 # px / um
    metadata = {'spacing' : 1 / camera_resolution,
                    'medium_index' : 1.348,
                    'illum_wavelen' : .660,
                    'illum_polarization' : (1, 0)}
    position = [640, 306]

    holonums = range(51)
    zpos = np.linspace(25, -25, 51) - 2.5
    paths = ["data/Mixed-60xWater-011019/greyscale-PS/im_" + "{}".format(str(num)).rjust(3, '0') + ".png"
             for num in holonums]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        holos = [mlf.load_bgdivide_crop(path=path, metadata=metadata,
                                        particle_position=position,
                                        bg_prefix="bkg",
                                        df_prefix="dark")
                 for path in paths]
    return holos, zpos

def load_few_BigPS_data_Jan10():
    camera_resolution = 5.9633 # px / um
    metadata = {'spacing' : 1 / camera_resolution,
                    'medium_index' : 1.348,
                    'illum_wavelen' : .660,
                    'illum_polarization' : (1, 0)}
    position = [901, 271]

    holonums = range(51)
    zpos = np.linspace(-25, 25, 51)
    paths = ["data/Mixed-60xWater-011019/greyscale-PS-big/im_" + "{}".format(str(num)).rjust(3, '0') + ".png"
             for num in holonums]
    holos = [mlf.load_bgdivide_crop(path=path, metadata=metadata,
                                particle_position=position, bg_prefix="bkg", df_prefix="dark")
             for path in paths]
    return holos, zpos

def load_few_Si_data_Jan10():
    camera_resolution = 5.9633 # px / um
    metadata = {'spacing' : 1 / camera_resolution,
                    'medium_index' : 1.348,
                    'illum_wavelen' : .660,
                    'illum_polarization' : (1, 0)}
    position = [607, 339]

    holonums = range(51)
    zpos = np.linspace(25, -25, 51) - 0.5
    paths = ["data/Mixed-60xWater-011019/greyscale-silica/im_" + "{}".format(str(num)).rjust(3, '0') + ".png"
             for num in holonums]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        holos = [mlf.load_bgdivide_crop(path=path, metadata=metadata,
                                        particle_position=position,
                                        bg_prefix="bkg", df_prefix="dark")
             for path in paths]
    return holos, zpos


def load_PS_data_Feb6():
    camera_resolution = 5.6983 # px / um
    metadata = {'spacing' : 1 / camera_resolution,
                    'medium_index' : 1.348,
                    'illum_wavelen' : .660,
                    'illum_polarization' : (1, 0)}
    position = [266, 680]

    holonums = np.arange(14, 137)
    zpos = np.linspace(13.5, -17, len(holonums))
    paths = ["data/Polystyrene-60xWater-020619/im" + "{}".format(str(num).rjust(4, '0')) + ".tif"
             for num in holonums]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        refimg = hp.load_image(paths[0], **metadata)
        bkg = mlf.load_bkg(paths[0], bg_prefix='bg', refimg=refimg)
        dark = mlf.load_dark(paths[0], df_prefix='dark', refimg=refimg)
        holos = [mlf.load_bgdivide_crop_v2(path=path, metadata=metadata,
                                        particle_position=position,
                                        bkg=bkg, dark=dark)
                 for path in paths]
    return holos, zpos

def load_PS_data_Feb6_alt():
    camera_resolution = 5.6983 # px / um
    metadata = {'spacing' : 1 / camera_resolution,
                    'medium_index' : 1.348,
                    'illum_wavelen' : .660,
                    'illum_polarization' : (1, 0)}
    position = [371, 860]

    holonums = range(134)
    zpos = np.linspace(17, -17, len(holonums))
    paths = ["data/Polystyrene-60xWater-020619/raw0/im" + "{}".format(str(num).rjust(4, '0')) + ".tif"
             for num in holonums]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        refimg = hp.load_image(paths[0], **metadata)
        bkg = mlf.load_bkg(paths[0], bg_prefix='bg', refimg=refimg)
        dark = mlf.load_dark(paths[0], df_prefix='dark', refimg=refimg)
        holos = [mlf.load_bgdivide_crop_v2(path=path, metadata=metadata,
                                        particle_position=position,
                                        bkg=bkg, dark=dark)
                 for path in paths]
    return holos, zpos

# def load_few_PS_sedemintation_data_Jan24():
#     camera_resolution = 5.6983 # px / um
#     metadata = {'spacing' : 1 / camera_resolution,
#                     'medium_index' : 1.348,
#                     'illum_wavelen' : .660,
#                     'illum_polarization' : (1, 0)}
#     position = [242, 266]

#     holonums = [19, 20]
#     zpos = np.linspace(25, -25, 51) - 0.5
#     paths = ["data/Polystyrene2-4um-60xWater-012419/raw/image" +  f"{num}".zfill(4) + ".png"
#              for num in holonums]
#     with warnings.catch_warnings():
#         warnings.simplefilter("ignore")
#         holos = [mlf.load_bgdivide_crop(path=path, metadata=metadata,
#                                         particle_position=position,
#                                         bg_prefix="bkg", df_prefix="dark")
#              for path in paths]
#     return holos, zpos

def load_PS_data_reza():
    resolution = 1/.194 # pixels per um
    metadata = {'spacing' : 1 / resolution,
                'medium_index' : 1.34,
                'illum_wavelen' : .660,
                'illum_polarization' : (1, 0)}
    PS_location = [258, 255]

    holonums = range(80)
    zpos = np.linspace(20, -20, 80)
    paths = ["data/Polystyrene-Reza/image" + "{}".format(str(num).rjust(4, '0')) + ".tif"
             for num in holonums]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        holos = [mlf.load_bgdivide_crop(path=path, metadata=metadata,
                                        particle_position=position,
                                        bg_prefix="bg")
             for path in paths]
    return holos, zpos

def make_guess_parameters(zpos, n, r):
    return [{'z': z, 'n': n, 'r': r} for z in zpos]

def hologram2array(hologram):
    return hologram.values.squeeze()

def compare_imgs(im1, im2, titles=['im1', 'im2']):
    vmax = np.max((im1, im2))
    vmin = np.min((im1, im2))

    plt.figure(figsize=(10,5))
    plt.gray()
    plt.subplot(121)
    plt.imshow(im1, interpolation="nearest", vmin=vmin, vmax=vmax)
    plt.colorbar()
    plt.title(titles[0])
    plt.subplot(122)
    plt.imshow(im2, interpolation="nearest", vmin=vmin, vmax=vmax)
    plt.colorbar()
    plt.title(titles[1])
    plt.show()


def compare_holos(*holos, titles=None, cmap="Greys"):
    ims = [holo.values.squeeze() for holo in holos]
    vmax = np.max(ims)
    vmin = np.min(ims)

    plt.figure(figsize=(5*len(ims),5))
    plt.gray()
    for index, im in enumerate(ims):
        plt.subplot(1, len(ims), index + 1)
        plt.imshow(im, interpolation="nearest", vmin=vmin, vmax=vmax, cmap=cmap)
        plt.colorbar()
        if titles:
            plt.title(titles[index])
    plt.show()

def fit_data():
    PS_data, PS_zpos = load_PS60xWater_data()
    Si_data, Si_zpos = load_Silica60xWater_data()

    PS_guess = make_guess_parameters(PS_zpos, n=1.58, r=0.8)
    Si_guess = make_guess_parameters(Si_zpos, n=1.46, r=0.5)

    PS_fit = [mlf.fit_mielens(data, guess) for data, guess in zip(PS_data, PS_guess)]
    Si_fit = [mlf.fit_mielens(data, guess) for data, guess in zip(Si_data, Si_guess)]

    PS_holo = [hologram2array(data) for data in PS_data]
    Si_holo = [hologram2array(data) for data in Si_data]

    PS_errsq = [mlf.calc_err_sq(data, fit.scatterer, **fit.parameters) for data, fit in zip(PS_data, PS_fit)]
    Si_errsq = [mlf.calc_err_sq(data, fit.scatterer, **fit.parameters) for data, fit in zip(Si_data, Si_fit)]

    Si_fit_holo = [hologram2array(calc_holo(data, fit.scatterer, theory=MieLens(lens_angle=fit.parameters['lens_angle'])))
                   for data, fit in zip(Si_data, Si_fit)]
    PS_fit_holo = [hologram2array(calc_holo(data, fit.scatterer, theory=MieLens(lens_angle=fit.parameters['lens_angle'])))
                   for data, fit in zip(PS_data, PS_fit)]

def compare_guess_holo(data, guesses):
    guess_scatterer, guess_lens_angle = mlf.get_guess_scatterer(data, guesses)
    guess_holo = hologram2array(calc_holo(data, guess_scatterer,
                                theory=MieLens(lens_angle=lens_angle)))
    data_holo = hologram2array(data)
    compare_imgs(data_holo, guess_holo, ['data', 'guess'])

def compare_fit_Jan4_data():
    PS_data, PS_zpos = load_few_PS_data_Jan4()
    PS_guess = make_guess_parameters(PS_zpos, n=1.58, r=0.5)

    mofit = [mlf.fit_mieonly(data, guess) for data, guess in zip(PS_data, PS_guess)]
    mlfit = [mlf.fit_mielens(data, guess) for data, guess in zip(PS_data, PS_guess)]

    moholo = [calc_holo(data, fit.scatterer, scaling=fit.parameters['alpha']) for data, fit in zip(PS_data, mofit)]
    mlholo = [calc_holo(data, fit.scatterer, theory=MieLens(lens_angle=fit.parameters['lens_angle'])) for data, fit in zip(PS_data, mlfit)]

    moerrsq = [mlf.calc_err_sq(data, fit.scatterer, **fit.parameters, theory='mieonly') for data, fit in zip(PS_data, mofit)]
    mlerrsq = [mlf.calc_err_sq(data, fit.scatterer, **fit.parameters) for data, fit in zip(PS_data, mlfit)]

def make_stack_figures(data, fits, n=None, r=None, z_positions=None):
    scatterers = [fit.scatterer for fit in fits]
    z = [fit.scatterer.center[2] for fit in fits] if z_positions is None else z_positions
    for scatterer, z_pos in zip(scatterers, z):
        scatterer.n = n if n is not None else scatterer.n
        scatterer.r = r if r is not None else scatterer.r
        scatterer.center[2] = z_pos if z_positions is not None else scatterer.center[2]

    model_holos = [calc_holo(dt, scatterer, theory=MieLens(lens_angle=1.0))
                   for dt, scatterer in zip(data, scatterers)]

    data_stack_xz = np.vstack([dt.values.squeeze()[50,:] for dt in data])
    data_stack_yz = np.vstack([dt.values.squeeze()[:,50] for dt in data])

    model_stack_xz = np.vstack([holo.values.squeeze()[50,:] for holo in model_holos])
    model_stack_yz = np.vstack([holo.values.squeeze()[:,50] for holo in model_holos])

    return data_stack_xz, data_stack_yz, model_stack_xz, model_stack_yz


if __name__ == '__main__':
    # Load PS data
    PS_data, PS_zpos = load_few_PS_data_Jan10()
    PS_guess = make_guess_parameters(PS_zpos, n=1.58, r=0.5)

    # Load fits I've already done
    mofit_PS = [
        hp.load("fits/PSJan10/mofit{num}.h5")
        for num in [
            "{}".format(str(num).rjust(3, '0'))
            for num in range(len(PS_data))]]
    mlfit_PS = [
        hp.load("fits/PSJan10/mlfit{num}.h5")
        for num in [
            "{}".format(str(num).rjust(3, '0'))
            for num in range(len(PS_data))]]
    moholo_PS = [
        calc_holo(data, fit.scatterer, scaling=fit.parameters['alpha']) for data, fit in zip(PS_data, mofit_PS)]
    mlholo_PS = [calc_holo(data, fit.scatterer, theory=MieLens(lens_angle=fit.parameters['lens_angle'])) for data, fit in zip(PS_data, mlfit_PS)]

    moerrsq_PS = [mlf.calc_err_sq(data, fit.scatterer, **fit.parameters, theory='mieonly') for data, fit in zip(PS_data, mofit_PS)]
    mlerrsq_PS = [mlf.calc_err_sq(data, fit.scatterer, **fit.parameters) for data, fit in zip(PS_data, mlfit_PS)]


    # Make figure 3 with this function
    data_stack_xz, data_stack_yz, model_stack_xz, model_stack_yz = make_stack_figures(PS_data, mlfit_PS, n=1.41, r=0.5, z_positions=np.linspace(26.5, -23.5, 51))

    compare_imgs(data_stack_xz, model_stack_xz)
    compare_imgs(data_stack_yz, model_stack_yz)

    # This is for Fitting the Silica data
    #mofit_Si = [mlf.fit_mieonly(data, guess) for data, guess in zip(Si_data, Si_guess)]
    #mlfit_Si = [mlf.fit_mielens(data, guess) for data, guess in zip(Si_data, Si_guess)]
    #moholo_Si = [calc_holo(data, fit.scatterer, scaling=fit.parameters['alpha']) for data, fit in zip(Si_data, mofit_Si)]
    #mlholo_Si = [calc_holo(data, fit.scatterer, theory=MieLens(lens_angle=fit.parameters['lens_angle'])) for data, fit in zip(Si_data, mlfit_Si)]

    #moerrsq_Si = [mlf.calc_err_sq(data, fit.scatterer, **fit.parameters, theory='mieonly') for data, fit in zip(Si_data, mofit_Si)]
    #mlerrsq_Si = [mlf.calc_err_sq(data, fit.scatterer, **fit.parameters) for data, fit in zip(Si_data, mlfit_Si)]
