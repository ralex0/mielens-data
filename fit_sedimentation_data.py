from collections import OrderedDict, namedtuple
from datetime import timedelta
import json
from multiprocessing import Pool
import pickle
import time

import numpy as np
import matplotlib.pyplot as plt
import holopy as hp
from lmfit import report_fit

import mielensfit as mlf  # 1.08 s
from lmfitFitter import Fitter  # 17.5 ms
import inout


__TIMER_CLICK__ = time.time()


def test_fit(dt, guess):
    fitter_ml = Fitter(theory='mielens')
    fitter_mo = Fitter(theory='mieonly')

    print("Fitting with mielens...")
    t0 = time.time()
    ml_fit = fitter_ml.fit(dt, guess)
    ml_time = timedelta(seconds=(time.time() - t0))
    print("Done! Took {} s".format(ml_time))
    print(report_fit(ml_fit))

    print("Fitting with mieonly...")
    mo_fit = fitter_mo.fit(dt, guess)
    mo_time = timedelta(seconds=(time.time() - t0 - ml_time.total_seconds()))
    print("Done! Took {} s".format(mo_time))
    print(report_fit(mo_fit))

    mo_holo = fitter_mo._calc_model(mo_fit.params, dt)
    ml_holo = fitter_ml._calc_model(ml_fit.params, dt)

    compare_holos(dt, mo_holo, ml_holo)

    return mo_fit, ml_fit


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


def _guesses_from_old(fits):
    return list(fits.values())


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


def cascade_fit(data, first_guess, fitter, particle, bottom_frame):
    guess = first_guess
    fits = []
    for d in data:
        next_fit = fitter.fit(d, guess)
        fits.append(next_fit)
        if fitter.theory == 'mieonly':
            keys = ['z', 'n', 'r', 'alpha']
        if fitter.theory == 'mielens':
            keys = ['z', 'n', 'r', 'lens_angle']
        guess = {k: next_fit.params[k] for k in keys}
    return fits

Particle = namedtuple("Particle", ["radius", "density"])
# We take the radii as the median radii from mielens. We use the
# median and not the mean to avoid the bad-fit outliers.
SILICA_PARTICLE = Particle(radius=0.7826, density=2.0)
# ..except the radii aren't correct for ml PS, so we use median(mo)
# for the PS
POLYSTYRENE_PARTICLE = Particle(radius=1.168, density=1.05)
VISCOSITY_WATER = 8.9e-4  # in mks units = Pa*s

def _calc_particle_trajectory(
        times, particle, initial_z_position, bottom_frame=None):
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
    trajectory = initial_z_position - times * velocity_microns_per_second

    # 3. Clip trajectory after it hits the coverslip
    if bottom_frame is not None:
        trajectory[bottom_frame:] = trajectory[bottom_frame]

    return trajectory


def fit_si_data():
    tick_tock()
    si_data = inout.load_silica_sedimentation_data()[0]
    si_times = np.load("./fits/sedimentation/Si_frame_times.npy")
    print("Time to load Si data: {}".format(tick_tock()))

    old_si_fits_mo, old_si_fits_ml = inout.load_silica_sedimentation_fits()
    si_initial_z = old_si_fits_ml['0']['z']
    si_bottom_frame = 60
    si_zpos = _calc_particle_trajectory(
        si_times, SILICA_PARTICLE, si_initial_z, si_bottom_frame)
    si_guesses = [
        {'n': 1.43, 'r': SILICA_PARTICLE.radius, 'z': z} for z in si_zpos]

    fitter_ml = Fitter(theory='mielens')
    fitter_mo = Fitter(theory='mieonly')
    try:
        pool = Pool(4)
        tick_tock()
        si_fits_mo = pool.starmap(
            fitter_mo.fit, zip(si_data[:43], si_guesses[:43]))
        si_fits_ml = pool.starmap(
            fitter_ml.fit, zip(si_data, si_guesses))
        print("Time to fit Si: {}".format(tick_tock()))
    finally:
        pool.close()
        pool.join()
    return si_fits_mo, si_fits_ml


def fit_ps_data():
    tick_tock()
    ps_data = inout.load_polystyrene_sedimentation_data()[0]
    ps_times = np.load("./fits/sedimentation/PS_frame_times.npy")
    print("Time to load PS data: {}".format(tick_tock()))

    old_ps_fits_mo, old_ps_fits_ml = inout.load_polystyrene_sedimentation_fits()
    ps_initial_z = old_ps_fits_ml['0']['z']
    ps_bottom_frame = 31 #Looks like it hits the coverslip around here
    ps_zpos = _calc_particle_trajectory(
        ps_times, POLYSTYRENE_PARTICLE, ps_initial_z, ps_bottom_frame)
    ps_guesses = [
        {'n': 1.58, 'r': POLYSTYRENE_PARTICLE.radius, 'z': z} for z in ps_zpos]

    fitter_ml = Fitter(theory='mielens')
    fitter_mo = Fitter(theory='mieonly')
    try:
        pool = Pool(4)
        tick_tock()
        ps_fits_mo = pool.starmap(
            fitter_mo.fit, zip(ps_data[:20], ps_guesses[:20]))
        ps_fits_ml = pool.starmap(
            fitter_ml.fit, zip(ps_data, ps_guesses))
        print("Time to fit PS: {}".format(tick_tock()))
    finally:
        pool.close()
        pool.join()
    return ps_fits_mo, ps_fits_ml


def save_results(ps_fits_mo, ps_fits_ml, si_fits_mo, si_fits_ml):
    si_params_ml = OrderedDict()
    savefolder = "fits/sedimentation/best_of_03-27_and_04-02/"
    for i, fit in enumerate(si_fits_ml):
        si_params_ml.update({str(i): fit.params.valuesdict()})
        pickle_filename = os.path.join(
            savefolder, "Si_mielens/{}.pkl".format(inout.zfill(i)))
        inout.save_pickle(fit, pickle_filename)

    json_filename = os.path.join(
        savefolder, 'mielens_sedimentation_params_Si.json')
    with open(json_filename, 'w') as f:
        json.dump(si_params_ml, f, indent=4)

    si_params_mo = OrderedDict()
    for i, fit in enumerate(si_fits_mo):
        si_params_mo.update({str(i): fit.params.valuesdict()})
        inout.save_pickle(fit,
            "fits/sedimentation/best_of_03-27_and_04-02/Si_mieonly/{}.pkl".format(inout.zfill(i)))
    with open('fits/sedimentation/best_of_03-27_and_04-02/mieonly_sedimentation_params_Si.json', 'w') as f:
        json.dump(si_params_mo, f, indent=4)

    ps_params_ml = OrderedDict()
    for i, fit in enumerate(si_fits_mo):
        ps_params_ml.update({str(i): fit.params.valuesdict()})
        inout.save_pickle(fit,
            "fits/sedimentation/best_of_03-27_and_04-02/PS_mielens/{}.pkl".format(inout.zfill(i)))
    with open('fits/sedimentation/best_of_03-27_and_04-02/mielens_sedimentation_params_PS.json', 'w') as f:
        json.dump(ps_params_ml, f, indent=4)

    ps_params_mo = OrderedDict()
    for i, fit in enumerate(ps_fits_mo):
        ps_params_mo.update({str(i): fit.params.valuesdict()})
        inout.save_pickle(fit,
            "fits/sedimentation/best_of_03-27_and_04-02/PS_mieonly/{}.pkl".format(inout.zfill(i)))
    with open('fits/sedimentation/best_of_03-27_and_04-02/mieonly_sedimentation_params_PS.json', 'w') as f:
        json.dump(ps_params_mo, f, indent=4)


if __name__ == '__main__':
    # ps_fits_mo, ps_fits_ml = fit_ps_data()
    # si_fits_mo, si_fits_ml = fit_si_data()
    # save_results(ps_fits_mo, ps_fits_ml, si_fits_mo, si_fits_ml)
    pass

