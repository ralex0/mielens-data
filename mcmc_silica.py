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


# TODO:
# Time mielens with npixels=1e2 vs npixels=1e3 vs npixels=1e4
if __name__ == '__main__':
    WHICH_IMAGE = 10
    WHICH_FIT = str(WHICH_IMAGE)
    data, zpos = inout.load_silica_sedimentation_data(size=140)
    fits_mo, fits_ml = inout.load_silica_sedimentation_params("04-02")

    mielensFitter = Fitter(theory='mielens')
    mieonlyFitter = Fitter(theory='mieonly')


    # mcmc_kws = {'burn': 0, 'steps': 2, 'nwalkers': 100,
    #            'thin': 1, 'workers': 4, 'ntemps': 5}

    npixels = data[WHICH_IMAGE].values.size
    # npixels = 10000
    # 1 step: 55 s
    # 2 steps: 1:24
    # So at 29 s / step.
    # I want this to finish by tomorrow at 6 am = 22 hours from now.
    # So that's max 22 * 60 * 2 =  2640 steps.
    # So we do 2600 steps
    mcmc_kws = {'burn': 0, 'steps': 2600, 'nwalkers': 100,
                'thin': 1, 'workers': 16, 'ntemps': 10}
    """
    print("Starting mieonly mcmc")
    tick_tock()
    optimization_result = mieonlyFitter.mcmc(
        guess_mo, data[1], mcmc_kws=mcmc_kws, npixels=npixels)
    print("mieonly mcmc took {}".format(tick_tock()))
    mcmc_mo = optimization_result['mcmc_result']
    fit_mo = optimization_result['lmfit_result']
    best_mo = optimization_result['best_result']
    inout.save_pickle(fit_mo, 'si_fit_mo_pt.pkl')
    inout.save_pickle(mcmc_mo, 'si_mcmc_mo_pt.pkl')
    inout.save_pickle(best_mo, 'si_best_mo_pt.pkl')
    inout.save_pickle(fit_mo, 'si_fit_mo_pt.pkl')
    """

    guess_mo = fits_mo[WHICH_FIT]
    guess_ml = fits_ml[WHICH_FIT]
    print("Starting mielens mcmc")
    tick_tock()
    optimization_result = mielensFitter.mcmc(
        guess_ml, data[WHICH_IMAGE], mcmc_kws=mcmc_kws, npixels=npixels)

    print("mielens mcmc took {}".format(tick_tock()))
    # With 10,000 px (all of them), it takes 24 s / step to run MCMC,
    # with 5 temps, 100 walkers, 4 workers
    mcmc_ml = optimization_result['mcmc_result']
    fit_ml = optimization_result['lmfit_result']
    best_ml = optimization_result['best_result']
    inout.save_pickle(fit_ml, 'si_fit_ml_pt-data2.pkl')
    inout.save_pickle(mcmc_ml, 'si_mcmc_ml_pt-data2.pkl')
    inout.save_pickle(best_ml, 'si_best_ml_pt-data2.pkl')

