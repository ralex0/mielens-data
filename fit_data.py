import os
import time
import warnings

import numpy as np

import matplotlib.pyplot as plt
from matplotlib import rc

from mpl_toolkits import mplot3d

import holopy as hp
from holopy.scattering import calc_holo
from holopy.scattering.theory import MieLens

import mielensfit as mlf


HERE = os.path.dirname(__file__)

def load_few_PS_data_Jan10():
    camera_resolution = 5.9633 # px / um
    metadata = {'spacing' : 1 / camera_resolution,
                'medium_index' : 1.348,
                'illum_wavelen' : .660,
                'illum_polarization' : (1, 0)}
    position = [640, 306]

    holonums = range(51)
    zpos = np.linspace(25, -25, 51) - 2.5
    paths = [HERE + "/data/Mixed-60xWater-011019/greyscale-PS/im_" +
             "{}".format(str(num)).rjust(3, '0') + ".png"
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


def compare_holos(*holos, titles=None, cmap="gray"):
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


def compare_guess_holo(data, guess):
    fitter = mlf.Fitter(data, guess)
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


def fit_with_previous_as_guess(data, first_guess):
    # 1. Fit the first point.
    first_fit = mlf.fit_mielens(data[0], first_guess)
    current_guess = {k: v for k, v in first_guess.items()}
    if 'lens_angle' not in current_guess:
        current_guess.update({'lens_angle': mlf.Fitter._default_lens_angle})

    def update_guess(fit):
        current_guess['n'] = fit.parameters['n']
        current_guess['r'] = fit.parameters['r']
        current_guess['z'] = fit.parameters['center.2']
        current_guess['lens_angle'] = fit.parameters['lens_angle']

    update_guess(first_fit)
    all_fits = [first_fit]
    for datum in data[1:]:
        try:
            this_fit = mlf.fit_mielens(datum, current_guess)
            update_guess(this_fit)
        except:
            this_fit = None
        all_fits.append(this_fit)
    return all_fits


def fit_from_scratch(data, guess):
    fits = []
    for num, (data, guess) in enumerate(zip(data, guesses)):
        try:
            result = mlf.fit_mielens(data, guess)
        except:
            result = None
        fits.append(result)
    return fits


def calculate_models(data, fits):
    fitholos = [
        calc_holo(
            datum, fit.scatterer,
            theory=MieLens(lens_angle=fit.parameters['lens_angle']))
        if fit is not None else 0 * datum + 1
        for datum, fit in zip(data, fits)]
    return fitholos

def load_Si_sedemintation_data_Feb15():
    camera_resolution = 5.6983 # px / um
    metadata = {'spacing' : 1 / camera_resolution,
                    'medium_index' : 1.33,
                    'illum_wavelen' : .660,
                    'illum_polarization' : (1, 0)}
    position = [553, 725]

    holonums = range(100)
    zpos = np.linspace(38, -12, 100)
    paths = ["data/Silica1um-60xWater-021519/raw/image" 
             +  f"{num}".zfill(4) + ".tif" for num in holonums] 
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        refimg = hp.load_image(paths[0], **metadata)
        bkg = mlf.load_bkg("data/Silica1um-60xWater-021519/raw/bg/", 
                           bg_prefix='bg', refimg=refimg)
        dark = mlf.load_dark("data/Silica1um-60xWater-021519/raw/dark/",
                             df_prefix='dark', refimg=refimg)
        holos = [mlf.load_bgdivide_crop_v2(path=path, metadata=metadata, 
                                        particle_position=position, 
                                        bkg=bkg, dark=dark, size=140) 
                 for path in paths]
    return holos, zpos


def load_few_PS_sedemintation_data_Jan24():
    camera_resolution = 5.6983 # px / um
    metadata = {'spacing' : 1 / camera_resolution,
                    'medium_index' : 1.33,
                    'illum_wavelen' : .660,
                    'illum_polarization' : (1, 0)}
    position = [242, 266]

    holonums = range(50)
    zpos = np.linspace(20, -12, 50)
    paths = ["data/Polystyrene2-4um-60xWater-012419/raw/image" 
             +  f"{num}".zfill(4) + ".tif" for num in holonums] 
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        holos = [mlf.load_bgdivide_crop(path=path, metadata=metadata, 
                                        particle_position=position, 
                                        bg_prefix="bg", df_prefix="dark") 
             for path in paths]
    return holos, zpos

class TrackingSedimentationFigure(object):
    def __init__(self, data, mielens_fits, mieonly_fits, frame_times=None):
        self.data = data
        self.mielens_fits = mielens_fits
        self.mieonly_fits = mieonly_fits
        self.frame_times = self._setup_frame_times(frame_times)
        self._setup_rcparams()

    def _setup_frame_times(self, frame_times):
        if frame_times is None:
            frame_times = range(len(self.mielens_fits))
        return frame_times

    def _setup_rcparams(self):
        rc('text', usetex=True)
        rc('font',**{'family': 'Times New Roman'})

    def make_figure(self, holonums):
        assert len(holonums) == 3
        fig = plt.figure(figsize = (5.25, 5.0))
        hologram_axes, sedimentation_axes, parameter_axes = self._make_axes(fig)
        self._plot_holograms(hologram_axes, holonums)
        self._plot_sedimentation(sedimentation_axes)
        self._plot_parameters(parameter_axes)
        return fig

    def _make_axes(self, fig):
        ax_topholo = fig.add_axes([0.0, 0.7, 1 / 5.25, 1 / 5.25], label="topholo")
        ax_midholo = fig.add_axes([0.0, 0.4, 1 / 5.25, 1 / 5.25], label="midholo")
        ax_btmholo = fig.add_axes([0.0, 0.1, 1 / 5.25, 1 / 5.25], label="midholo")
        ax_sedplt = fig.add_axes([1.125/5.25, 0.125/5.25, 2/5.25, 5.0/5.25], 
                                 projection='3d', label="sedplot")
        ax_n = fig.add_axes([3.5/5.25, 0.0, 1.75/5.25, 1.5/5.25], label="nplot")
        ax_r = fig.add_axes([3.5/5.25, 0.35, 1.75/5.25, 1.5/5.25], label="rplot")
        ax_z = fig.add_axes([3.5/5.25, 0.7, 1.75/5.25, 1.5/5.25], label="zplot")
        return [ax_topholo, ax_midholo, ax_btmholo], ax_sedplt, [ax_n, ax_r, ax_z]

    def _plot_holograms(self, axes, indices):
        holos = [hologram2array(self.data[num]) for num in indices]
        for num, ax in enumerate(axes):
            ax.imshow(holos[num], vmin=np.min(holos), vmax=np.max(holos), 
                        interpolation='nearest', cmap='gray')
            ax.axis('off')

    def _plot_sedimentation(self, axes):
        positions = {'x': [fit.parameters['center.0'] for fit in self.mielens_fits],
                     'y': [fit.parameters['center.1'] for fit in self.mielens_fits],
                     'z': [fit.parameters['center.2'] for fit in self.mielens_fits]}

        axes.plot3D(positions['x'], positions['y'], positions['z'], 'gray')
        axes.set_xlabel('x', {'size': 8})
        axes.set_xticklabels([])
        axes.set_ylabel('y', {'size': 8})
        axes.set_yticklabels([])
        axes.set_zlabel('z', {'size': 8})
        axes.set_title("Best Fit Position", {'size': 8})
        axes.set_aspect('equal', 'box')

    def _plot_parameters(self, axes):
        ax_n, ax_r, ax_z = axes
        ax_n.set_xlabel('Elapsed time (s)', {'size': 8})
        ax_n.set_ylabel('index of refraction', {'size': 8})
        ax_n.scatter(self.frame_times, [fit.scatterer.n for fit in self.mielens_fits], 
                     color='green', s=4, marker='o', label="with lens")
        ax_n.scatter(self.frame_times, [fit.scatterer.n for fit in self.mieonly_fits],
                     color='red', s=4, marker='^', label="without lens")
        ax_n.legend(fontsize=6)
        ax_n.tick_params(labelsize=7)

        ax_r.set_xlabel('Elapsed time (s)', {'size': 8})
        ax_r.set_ylabel('index of refraction', {'size': 8})
        ax_r.scatter(self.frame_times, [fit.scatterer.r for fit in self.mielens_fits],
                     color='green', s=4, marker='o', label="with lens")
        ax_r.scatter(self.frame_times, [fit.scatterer.r for fit in self.mieonly_fits],
                     color='red', s=4, marker='^', label="without lens")
        ax_r.legend(fontsize=6)
        ax_r.tick_params(labelsize=7)

        ax_z.set_xlabel('Elapsed time (s)', {'size': 8})
        ax_z.set_ylabel('z-position  ($\mu m$)', {'size': 8})
        ax_z.scatter(self.frame_times, [fit.scatterer.center[2] for fit in self.mielens_fits], 
                     color='green', s=4, marker='o', label="with lens")
        ax_z.scatter(self.frame_times, [fit.scatterer.center[2] for fit in self.mieonly_fits],
                     color='red', s=4, marker='^', label="without lens")
        ax_z.legend(fontsize=6)
        ax_z.tick_params(labelsize=7)


if __name__ == '__main__':
    Si_data = load_Si_sedemintation_data_Feb15()[0]
    PS_data = load_few_PS_sedemintation_data_Jan24()[0]
    PS_times = np.load("./fits/sedimentation/PS_frame_times.npy")
    Si_times = np.load("./fits/sedimentation/Si_frame_times.npy")

    mofit_Si = [hp.load(f"fits/sedimentation/Si_mieonly/{num}.h5") 
                for num in [f"{num}".zfill(3) for num in range(len(Si_data))]]
    mlfit_Si = [hp.load(f"fits/sedimentation/Si_mielens/{num}.h5") 
                for num in [f"{num}".zfill(3) for num in range(len(Si_data))]]

    mofit_PS = [hp.load(f"fits/sedimentation/PS_mieonly/{num}.h5") 
                for num in [f"{num}".zfill(3) for num in range(len(PS_data))]]
    mlfit_PS = [hp.load(f"fits/sedimentation/PS_mielens/{num}.h5") 
                for num in [f"{num}".zfill(3) for num in range(len(PS_data))]]

    figure_PS = TrackingSedimentationFigure(PS_data, mlfit_PS, mofit_PS, PS_times)
    figure_Si = TrackingSedimentationFigure(Si_data, mlfit_Si, mofit_Si, Si_times)

    fig = figure_Si.make_figure(holonums=[0, 50, 99])
    plt.show()

# if __name__ == '__main__':
#     # Load PS data
#     data, zpos = load_few_PS_data_Jan10()
#     guesses = make_guess_parameters(zpos, n=1.58, r=0.5)

#     fits_fromscratch = fit_from_scratch(data, guesses)
#     fits_fromprevious = fit_with_previous_as_guess(data, guesses[0])

#     # fit_fromscratch[-1] looks ok. fit_previous looks _awful_
#     # So we do this:
#     last_guess = {
#         'n': fits_fromscratch[-1].parameters['n'],
#         'z': fits_fromscratch[-1].parameters['center.2'],
#         'r': fits_fromscratch[-1].parameters['r'],
#         }
#     fits_fromnext = fit_with_previous_as_guess(data[::-1], last_guess)[::-1]

#     fitholos_fromscratch = calculate_models(data, fits_fromscratch)
#     fitholos_fromprevious = calculate_models(data, fits_fromprevious)
#     fitholos_fromnext = calculate_models(data, fits_fromnext)

#     residuals_fromscratch = [
#         data - model for data, model in zip(data, fitholos_fromscratch)]
#     chisqs_fromscratch = np.array(
#         [(r.values**2).sum() for r in residuals_fromscratch])

#     residuals_fromprevious = [
#         data - model for data, model in zip(data, fitholos_fromprevious)]
#     chisqs_fromprevious = np.array(
#         [(r.values**2).sum() for r in residuals_fromprevious])

#     residuals_fromnext = [
#         data - model for data, model in zip(data, fitholos_fromnext)]
#     chisqs_fromnext = np.array(
#         [(r.values**2).sum() for r in residuals_fromnext])

#     # Now we just pick the best one:
#     all_chisqs = {
#         'next': chisqs_fromnext,
#         'previous': chisqs_fromprevious,
#         'scratch': chisqs_fromscratch,
#         }
#     all_fits = {
#         'next': fits_fromnext,
#         'previous': fits_fromprevious,
#         'scratch': fits_fromscratch,
#         }
#     fits_best = []
#     for i in range(len(chisqs_fromnext)):
#         these_chisqs = {k: v[i] for k, v in all_chisqs.items()}
#         best_error = np.inf
#         best_key = ''
#         for key, error in these_chisqs.items():
#             if error < best_error:
#                 best_key = key
#                 best_error = error

#         fits_best.append(all_fits[best_key][i])

#     fitholos_best = calculate_models(data, fits_best)
#     residuals_best = [data - model for data, model in zip(data, fitholos_best)]
#     chisqs_best = np.array([(r.values**2).sum() for r in residuals_best])
#     # OH GOD SAVE THESE AS FAST AS POSSIBLE!!!
#     from collections import OrderedDict
#     import json
#     parameters = OrderedDict()
#     for i, f in enumerate(fits_best):
#         parameters.update({str(i): f.parameters})
#     json.dump(parameters, open('./good-fit-parameters.json', 'w'), indent=4)

