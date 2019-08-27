from datetime import timedelta
import os
import time

from lmfitFitter import Fitter
import inout


WHICH_SPHERE = 'silica'
SIZE = 256
THEORY = 'mielensalpha'
MCMC_KWS = {'burn': 0, 'steps': 1024, 'nwalkers': 128,
            'thin': 1, 'workers': 4, 'ntemps': 7}

__TIMER_CLICK__ = time.time()

def tick_tock():
    global __TIMER_CLICK__
    last_time =  __TIMER_CLICK__
    current_time  = __TIMER_CLICK__ = time.time()
    return timedelta(seconds=current_time - last_time)

def analyze_one_image(which_image):
    which_fit = str(which_image)
    data = inout.fastload_silica_sedimentation_data(size=SIZE)[which_image]
    fits_mo, fits_ml = inout.load_silica_sedimentation_params()

    if THEORY == 'mieonly':
        guess = fits_mo[which_fit]
    elif THEORY == 'mielens' or THEORY == 'mielensalpha':
        guess = fits_ml[which_fit]

    fitter = Fitter(theory=THEORY)
    # WTF? 10000 pixels takes 10 s / iteration,
    # 14400 px takes 48.74 s / iteration. So, 1e4 px:
    npixels = int(1e4)

    print("Starting {} mcmc".format(THEORY))
    tick_tock()
    optimization_result = fitter.mcmc(
        guess, data, mcmc_kws=MCMC_KWS, npixels=npixels)

    print("{} mcmc took {}".format(THEORY, tick_tock()))
    # With 10,000 px (all of them), it takes 24 s / step to run MCMC,
    # with 5 temps, 100 walkers, 4 workers
    mcmc_ml = optimization_result['mcmc_result']
    fit_ml = optimization_result['lmfit_result']

    best_ml = optimization_result['best_result']

    prefix = 'Silica021519-'
    prefix += '{}-{}-frame={}-size={}-npx={}'.format(
        WHICH_SPHERE, THEORY, which_image, SIZE, npixels)

    tosave = {'fit': fit_ml, 'mcmc': mcmc_ml, 'best': best_ml}
    for suffix, obj in tosave.items():
        filename = prefix + '-{}.pkl'.format(suffix)
        inout.save_pickle(obj, filename)


if __name__ == '__main__':
    for which_image in range(25):
        analyze_one_image(which_image)
