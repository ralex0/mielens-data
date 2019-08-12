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
    data_list, zpos = inout.load_silica_sedimentation_data(
        size=140, holonums=[WHICH_IMAGE])
    data = data_list[0]
    fits_mo, fits_ml = inout.load_silica_sedimentation_params("04-02")

    mielensalpha_fitter = Fitter(theory='mielensalpha')


    # mcmc_kws = {'burn': 0, 'steps': 2, 'nwalkers': 100,
    #            'thin': 1, 'workers': 4, 'ntemps': 5}

    npixels = data.values.size
    # npixels = 10000
    # 1 step: 55 s
    # 2 steps: 1:24
    # So at 29 s / step.
    # I want this to finish by tomorrow at 6 am = 22 hours from now.
    # So that's max 22 * 60 * 2 =  2640 steps.
    # So we do 2600 steps
    mcmc_kws = {'burn': 0, 'steps': 2, 'nwalkers': 16,
                'thin': 1, 'workers': 16, 'ntemps': 3}

    guess_mo = fits_mo[WHICH_FIT]
    guess_ml = fits_ml[WHICH_FIT]
    print("Starting mielens mcmc")
    tick_tock()
    optimization_result = mielensalpha_fitter.mcmc(
        guess_ml, data, mcmc_kws=mcmc_kws, npixels=npixels)

    print("mielens mcmc took {}".format(tick_tock()))
    # With 10,000 px (all of them), it takes 24 s / step to run MCMC,
    # with 5 temps, 100 walkers, 4 workers
