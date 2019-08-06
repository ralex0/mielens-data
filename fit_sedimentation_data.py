from collections import OrderedDict, namedtuple
from datetime import timedelta
from multiprocessing import Pool
import os
import time


import numpy as np

import matplotlib.pyplot as plt

import holopy as hp

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



def fit_ps_data():
    data = inout.fastload_polystyrene_sedimentation_data(size=256)
    guess_ml = inout.load_polystyrene_sedimentation_params_temp()

    fitter_ml = Fitter(theory='mieonly')

    tick_tock()
    print('Fiting mieonly...')
    with Pool(os.cpu_count()) as pool:
        fits_ml = pool.starmap(fitter_ml.fit,
                                   zip(data[:451], list(guess_ml.values()))[:451])
    print("Time to fit mieonly: {}".format(tick_tock()))
    return fits_ml

def _pick_best_fit(*fits):
    best_fit = fits[0]
    for fit in fits[1:]:
        if fit.chisqr < best_fit.chisqr:
            best_fit = fit
    return best_fit

if __name__ == '__main__':
    fits_mo = fit_ps_data()
    fits_dict = {"{}".format(num): fit.params.valuesdict() for num, fit in enumerate(fits_mo)}
    prefix = '/n/manoharan/alexander/mielens-data/fits/Polystyrene2-4um-60xWater-042919/'
    inout.save_json(fits_dict, prefix + 'mieonly_fits.json')
