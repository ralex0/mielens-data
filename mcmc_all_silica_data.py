from datetime import timedelta
import time

from lmfitFitter import Fitter
import inout


WHICH_SPHERE = 'silica'
SIZE = 250
THEORY = 'mieonly'
MCMC_KWS = {'burn': 0, 'steps': 1000, 'nwalkers': 100,
            'thin': 1, 'workers': 16, 'ntemps': 7}


def analyze_one_image(which_image):
    which_fit = str(which_image)
    if WHICH_SPHERE == 'polystyrene':
        data_list, zpos = inout.load_polystyrene_sedimentation_data(
            size=SIZE, holonums=[which_image])
        fits_mo, fits_ml = inout.load_polystyrene_sedimentation_params("03-27")
    elif WHICH_SPHERE == 'silica':
        data_list, zpos = inout.load_silica_sedimentation_data(
            size=SIZE, holonums=[which_image])
        fits_mo, fits_ml = inout.load_silica_sedimentation_params("04-02")
    data = data_list[0]

    if THEORY == 'mieonly':
        guess = fits_mo[which_fit]
    elif THEORY == 'mielens':
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

    prefix = '{}-{}-frame={}-size={}-npx={}'.format(
        WHICH_SPHERE, THEORY, which_image, SIZE, npixels)

    tosave = {'fit': fit_ml, 'mcmc': mcmc_ml, 'best': best_ml}
    for suffix, obj in tosave.items():
        filename = prefix + '-{}.pkl'.format(suffix)
        inout.save_pickle(obj, filename)



# TODO:
# Time mielens with npixels=1e2 vs npixels=1e3 vs npixels=1e4
if __name__ == '__main__':
    for which_image in [8, 9, 11, 12, 13]:
        analyze_one_image(which_image)

