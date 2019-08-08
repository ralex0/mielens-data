from collections import OrderedDict, namedtuple
from datetime import timedelta
from multiprocessing import Pool
import os
import sys
import time

import numpy as np

import matplotlib.pyplot as plt

import holopy as hp
from holopy.scattering import calc_holo, Sphere
from holopy.scattering.theory import MieLens

from lmfit import report_fit

from lmfitFitter import Fitter  # 17.5 ms
import inout


__TIMER_CLICK__ = time.time()

def compare_holos(*holos, titles=None, cmap="gray"):
    ims = [holo.values.squeeze() for holo in holos]
    vmax = np.max(ims)
    vmin = np.min(ims)

    plt.figure(figsize=(5*len(ims),5))
    plt.gray()
    for index, im in enumerate(ims):
        plt.subplot(1, len(ims), index + 1)
        plt.imshow(
            im, interpolation="nearest", vmin=vmin, vmax=vmax, cmap=cmap)
        plt.colorbar()
        if titles:
            plt.title(titles[index])
    plt.show()


def tick_tock():
    global __TIMER_CLICK__
    last_time =  __TIMER_CLICK__
    current_time  = __TIMER_CLICK__ = time.time()
    return timedelta(seconds=current_time - last_time)


def clip_data_to(fits, max_frame):
    clipped_data = OrderedDict()
    for k, v in fits.items():
        if int(k) <= max_frame:
            clipped_data.update({k: v})
    return clipped_data


def cascade_fit(data, first_guess, fitter):
    guess = first_guess
    fits = []
    for d in data:
        next_fit = fitter.fit(d, guess)
        fits.append(next_fit)
        if fitter.theory == 'mieonly':
            keys = ['z', 'n', 'r', 'alpha']
        if fitter.theory == 'mielens':
            keys = ['z', 'n', 'r', 'lens_angle']
        if fitter.theory == 'mielensalpha':
            keys = ['z', 'n', 'r', 'lens_angle', 'alpha']
        guess = {k: next_fit.params[k] for k in keys}
    return fits

Particle = namedtuple("Particle", ["radius", "density"])
# We take the radii as the median radii from mielens. We use the
# median and not the mean to avoid the bad-fit outliers.
SILICA_PARTICLE = Particle(radius=0.7826, density=2.0)
# ..except the radii aren't correct for ml PS, so we use median(mo)
# for the PS
POLYSTYRENE_PARTICLE = Particle(radius=1.2, density=1.05)
VISCOSITY_WATER = 8.9e-4  # in mks units = Pa*s

def _calc_particle_trajectory(
        numframes, particle, initial_z_position, bottom_frame=None):
    # 1. Calculate the velocity, using meter-kilogram-second units:
    radius = particle.radius * 1e-6
    density = (particle.density - 1) * 1e3  # 1 g / cc = 1e3 kg / m^3
    volume = 4 * np.pi / 3. * radius**3
    mass = density * volume
    gravity = 9.8  # mks
    force = mass * gravity
    drag = 6 * np.pi * VISCOSITY_WATER * radius
    velocity_meters_per_second = force / drag
    velocity_microns_per_second = 1e6 * velocity_meters_per_second

    # 2. Calculate the trajectory:
    framerate = FRAMERATE_PS_DATA
    total_time = numframes/framerate
    times = np.linspace(0, total_time, numframes)
    trajectory = initial_z_position - times * velocity_microns_per_second

    # 3. Clip trajectory after it hits the coverslip
    if bottom_frame is not None:
        trajectory[bottom_frame:] = trajectory[bottom_frame]

    return trajectory

def _pick_best_fit(*fits):
    best_fit = fits[0]
    for fit in fits[1:]:
        if fit.chisqr < best_fit.chisqr:
            best_fit = fit
    return best_fit

def run_fits(particle, theory):
    if particle == 'polystyrene':
        data = inout.fastload_polystyrene_sedimentation_data()
        guesses = inout.load_polystyrene_sedimentation_guesses()
        if theory == 'mieonly':
            data = data[:451]
            guesses = guesses[:451]
    elif partcle == 'silica':
        data = inout.fastload_silica_sedimentation_data()
        guesses = inout.load_silica_sedimentation_guesses()
        if theory == 'mieonly':
            data = data[:451]
            guesses = guesses[:451]
    return fit_data(data, guesses, theory)

def fit_data(data, guesses, theory='mielensalpha'):
    fitter = Fitter(theory=theory)
    tick_tock()
    print('Fiting {}...'.format(theory))
    with Pool(os.cpu_count()) as pool:
        fits = pool.starmap(fitter.fit, zip(data, guesses))
    print("Time to fit {}: {}".format(theory, tick_tock()))
    return fits

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

def holo_from_fit(fit, refimg):
    center = (fit['x'], fit['y'], fit['z'])
    r = fit['r']
    n = fit['n']
    alpha = fit['alpha']
    sph = Sphere(n=n, r=r, center=center)
    theory = MieLens(lens_angle=fit['lens_angle']) if 'lens_angle' in fit else 'auto'
    return calc_holo(refimg, sph, scaling=alpha, theory=theory)

if __name__ == '__main__':
    particle = str(sys.argv[1]).lower()
    theory = str(sys.argv[2]).lower()
    fits = run_fits(particle, theory)
    fits_dict = {"{}".format(num): fit.params.valuesdict()
                 for num, fit in enumerate(fits)}
    prefix = '/n/manoharan/alexander/mielens-data/fits/'
    inout.save_json(fits_dict, prefix + '{}_{}_fits.json'.format(particle, theory))
    # fitter = Fitter(theory='mielensalpha')
    # data = inout.fastload_silica_sedimentation_data()[0]
    # guess = {'n': 1.43, 'r': 0.5, 'z': 20.}
    # tick_tock()
    # fit = fitter.fit(data, guess)
    # print(f"fit took {tick_tock()}.")
    # holo = holo_from_fit(fit.params.valuesdict(), refimg=data)
    # compare_imgs(data.values.squeeze(), holo.values.squeeze())
