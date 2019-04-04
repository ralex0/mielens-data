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


if __name__ == '__main__':
    data, zpos = inout.load_polystyrene_sedimentation_data()
    fits_mo, fits_ml = inout.load_polystyrene_sedimentation_params("03-27")

    mielensFitter = Fitter(theory='mielens')
    mieonlyFitter = Fitter(theory='mieonly')

    guess_mo = fits_mo['1']
    guess_ml = fits_ml['1']

    mcmc_kws = {'burn': 0, 'steps': 1000, 'nwalkers': 100, 
                'thin': 1, 'workers': 4, 'ntemps': 5}

    print("Starting mieonly mcmc")
    tick_tock()
    mcmc_mo, fit_mo = mieonlyFitter.mcmc(guess_mo, data[1], mcmc_kws=mcmc_kws, 
                                         npixels=100)
    print(f"mieonly mcmc took {tick_tock()}")
    inout.save_pickle(fit_mo, 'ps_fit_mo_pt.pkl')
    inout.save_pickle(mcmc_mo, 'ps_mcmc_mo_pt.pkl')

    print("Starting mielens mcmc")
    tick_tock()
    mcmc_ml, fit_ml = mielensFitter.mcmc(guess_ml, data[1], mcmc_kws=mcmc_kws,
                                         npixels=100)
    print(f"mielens mcmc took {tick_tock()}")
    inout.save_pickle(fit_ml, 'ps_fit_ml_pt.pkl')
    inout.save_pickle(mcmc_ml, 'ps_mcmc_ml_pt.pkl')

    
    