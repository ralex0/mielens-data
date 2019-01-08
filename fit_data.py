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
    paths = ["data/Silica-100xOil-120418/raw/im_" +  f"{num}".zfill(3) + ".png"
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
    paths = ["data/Polystyrene-100xOil-120418/greyscale/im" +  f"{num}".zfill(3) + ".png"
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
    paths = ["data/Polystyrene-60xWater-111918/raw/im" +  f"{num}".zfill(3) + ".png"
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
    paths = ["data/Polystyrene-60xWater-121218/raw/im_" +  f"{num}".zfill(3) + ".png"
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
    paths = ["data/Silica-60xWater-121218/raw/im_" +  f"{num}".zfill(3) + ".png"
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
    paths = ["data/Polystyrene-60xWater-010419/raw0_greyscale/im" +  f"{num}".zfill(3) + ".png"
             for num in holonums] 
    holos = [mlf.load_bgdivide_crop(path=path, metadata=metadata, 
                                particle_position=position, bg_prefix="bg", df_prefix="dark") 
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

if __name__ == '__main__':
    PS_data, PS_zpos = load_few_PS_data_Jan4()
    PS_guess = make_guess_parameters(PS_zpos, n=1.58, r=0.5)

    data = PS_data[2]
    guess = PS_guess[2]
    guess['z'] = -10
    fit = mlf.fit_mielens(data, guess)
    gop = mlf.globalop_mielens(data, guess)
    polished_guess = {'z': gop.parameters['center.2'], 'n': gop.parameters['n'], 'r': gop.parameters['r']}
    polished_gop = mlf.fit_mielens(data, polished_guess)

    fit_err = mlf.calc_err_sq(data, fit.scatterer, **fit.parameters, theory='mielens')
    gop_err = mlf.calc_err_sq(data, gop.scatterer, **gop.parameters, theory='mielens')
    pgop_err = mlf.calc_err_sq(data, polished_gop.scatterer, **polished_gop.parameters, theory='mielens')

    # mofit = [mlf.fit_mieonly(data, guess) for data, guess in zip(PS_data, PS_guess)]
    # mlfit = [mlf.fit_mielens(data, guess) for data, guess in zip(PS_data, PS_guess)]

    # moholo = [calc_holo(data, fit.scatterer, scaling=fit.parameters['alpha']) for data, fit in zip(PS_data, mofit)]   
    # mlholo = [calc_holo(data, fit.scatterer, theory=MieLens(lens_angle=fit.parameters['lens_angle'])) for data, fit in zip(PS_data, mlfit)]
    
    # moerrsq = [mlf.calc_err_sq(data, fit.scatterer, **fit.parameters, theory='mieonly') for data, fit in zip(PS_data, mofit)]
    # mlerrsq = [mlf.calc_err_sq(data, fit.scatterer, **fit.parameters) for data, fit in zip(PS_data, mlfit)]