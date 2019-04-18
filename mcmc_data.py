from datetime import timedelta
import time

from lmfitFitter import Fitter
import inout


__TIMER_CLICK__ = time.time()
def tick_tock():
    global __TIMER_CLICK__
    last_time =  __TIMER_CLICK__
    current_time  = __TIMER_CLICK__ = time.time()
    return timedelta(seconds=(current_time - last_time))


def time_mcmc(
        data, initial_guesses, theory='mielens', mcmc_kws=None, npixels=100):
    if mcmc_kws is None:
        mcmc_kws = {'burn': 0, 'steps': 1, 'nwalkers': 100,
                    'thin': 1, 'workers': 16, 'ntemps': 7}
    fitter = Fitter(theory=theory)
    start_fit = time.time()
    best_fit = fitter.fit(data, initial_guesses)
    end_fit = time.time()

    copied_kwargs = mcmc_kws.copy()
    mcmc_times = {}
    for steps in [1, 10]:
        start_mcmc = time.time()
        copied_kwargs.update({'steps': steps})
        _ = fitter._mcmc(
            best_fit, data, mcmc_kws=copied_kwargs, npixels=npixels)
        end_mcmc = time.time()
        mcmc_times.update({steps: end_mcmc - start_mcmc})
    print("Fitting time:\t{:.3} s".format(end_fit - start_fit))
    for steps, times in mcmc_times.items():
        print("{} MCMC steps:\t{:.3} s".format(steps, times))


# TODO:
# Time mielens with npixels=1e2 vs npixels=1e3 vs npixels=1e4
if __name__ == '__main__':
    # WHICH_SPHERE = 'polystyrene'
    WHICH_SPHERE = 'silica'
    WHICH_IMAGE = 2
    WHICH_FIT = str(WHICH_IMAGE)
    SIZE = 250
    THEORY = 'mielens'

    if WHICH_SPHERE == 'polystyrene':
        data_list, zpos = inout.load_polystyrene_sedimentation_data(
            size=SIZE, holonums=[WHICH_IMAGE])
        fits_mo, fits_ml = inout.load_polystyrene_sedimentation_params("03-27")
    elif WHICH_SPHERE == 'silica':
        data_list, zpos = inout.load_silica_sedimentation_data(
            size=SIZE, holonums=[WHICH_IMAGE])
        fits_mo, fits_ml = inout.load_silica_sedimentation_params("04-02")
    data = data_list[0]

    if THEORY == 'mieonly':
        guess = fits_mo[WHICH_FIT]
    elif THEORY == 'mielens':
        guess = fits_ml[WHICH_FIT]

    fitter = Fitter(theory=THEORY)
    mcmc_kws = {'burn': 0, 'steps': 5000, 'nwalkers': 100,
                'thin': 1, 'workers': 16, 'ntemps': 7}

    # WTF? 10000 pixels takes 10 s / iteration,
    # 14400 px takes 48.74 s / iteration. So, 1e4 px:
    # npixels = int(1e4)
    # time_mcmc(
    #     data, guess, theory=THEORY, mcmc_kws=mcmc_kws, npixels=npixels)

    print("Starting {} mcmc".format(THEORY))
    tick_tock()
    optimization_result = fitter.mcmc(
        guess, data, mcmc_kws=mcmc_kws, npixels=npixels)

    print("{} mcmc took {}".format(THEORY, tick_tock()))
    # With 10,000 px (all of them), it takes 24 s / step to run MCMC,
    # with 5 temps, 100 walkers, 4 workers
    mcmc_ml = optimization_result['mcmc_result']
    fit_ml = optimization_result['lmfit_result']
    best_ml = optimization_result['best_result']

    prefix = 'polystyrene-{}-frame={}-size={}-npx={}'.format(
        THEORY, WHICH_IMAGE, SIZE, npixels)

    tosave = {'fit': fit_ml, 'mcmc': mcmc_ml, 'best': best_ml}
    for suffix, obj in tosave.items():
        filename = prefix + '-{}.pkl'.format(suffix)
        inout.save_pickle(obj, filename)

