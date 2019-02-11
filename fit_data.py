import time
import warnings

import numpy as np

import matplotlib.pyplot as plt

import holopy as hp
from holopy.scattering import calc_holo
from holopy.scattering.theory import MieLens

import mielensfit as mlf


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
        holos = mlf.load_bgdivide_crop_all_images(
            paths, metadata, position, darkfield_prefix="dark",
            background_prefix="bkg")
        # holos = [mlf.load_bgdivide_crop(
            # path=path, metadata=metadata, particle_position=position,
            # bg_prefix="bkg", df_prefix="dark")
            # for path in paths]
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
    ax1 = plt.subplot(1, 3, 1)
    plt.imshow(im1, interpolation="nearest", vmin=vmin, vmax=vmax)
    # plt.colorbar()
    plt.title(titles[0])
    ax2 = plt.subplot(1, 3, 2)
    plt.imshow(im2, interpolation="nearest", vmin=vmin, vmax=vmax)
    # plt.colorbar()
    plt.title(titles[1])

    difference = im1 - im2
    vmax = np.abs(difference).max()
    ax3 = plt.subplot(1, 3, 3)
    plt.imshow(difference, vmin=-vmax, vmax=vmax, interpolation='nearest',
               cmap='RdBu')
    chisq = np.sum(difference**2)
    plt.title("Difference, $\chi^2$={:0.2f}".format(chisq))

    for ax in [ax1, ax2, ax3]:
        ax.set_xticks([])
        ax.set_yticks([])

    plt.show()
    plt.tight_layout()


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


def compare_guess_holo(data, guesses):
    fitter = mlf.Fitter(data, guesses)
    guess_scatterer = fitter.make_guessed_scatterer()
    guess_lens_angle = fitter.guess_lens_angle()
    compare_fit_holo(data, guess_scatterer, guess_lens_angle.guess)


def compare_fit_holo(data, fit_scatterer, fit_lens_angle):
    guess_holo = hologram2array(
        calc_holo(data, fit_scatterer,
                  theory=MieLens(lens_angle=fit_lens_angle)))
    data_holo = hologram2array(data)
    compare_imgs(data_holo, guess_holo, ['Data', 'Model'])


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
    print("loading data...")
    PS_data, PS_zpos = load_few_PS_data_Jan10()
    print("guessing parameters...")
    PS_guess = make_guess_parameters(PS_zpos, n=1.58, r=0.5)

    # Fit the data:
    # ~~~ For speed we only fit the worst one, ``33`` with a chisq of 99.1
    PS_data = PS_data[33:34]
    PS_zpos = PS_zpos[33:34]
    PS_guess = PS_guess[33:34]
    # ~~~
    print("fitting data...")
    mlfit_PS = [mlf.fit_mielens(data, guess)
                for data, guess in zip(PS_data, PS_guess)]
    mlholo_PS = [
        calc_holo(
            data, fit.scatterer,
            theory=MieLens(lens_angle=fit.parameters['lens_angle']))
        for data, fit in zip(PS_data, mlfit_PS)]
    residuals = [data - model for data, model in zip(PS_data, mlholo_PS)]
    chisqs = [(r.values**2).sum() for r in residuals]

