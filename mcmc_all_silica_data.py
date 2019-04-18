"""
Idea: Mielens is ~N, where N is the # of px in the subset. So we just
drop the # of pixels down to 1000 and do parallel tempering with a decent
number of temperatures, to see if we can get better fits.
"""
from datetime import timedelta
import time

from lmfitFitter import Fitter
import inout

WHICH_SPHERE = 'silica'
SIZE = 250
THEORY = 'mieonly'
NPIXELS = int(1e3)
MCMC_KWS = {'burn': 0, 'steps': 3000, 'nwalkers': 100,
            'thin': 1, 'workers': 16, 'ntemps': 5}
FITTER = Fitter(theory=THEORY, quiet=True)


def time_mcmc(data, initial_guesses):
    start_fit = time.time()
    best_fit = FITTER.fit(data, initial_guesses)
    end_fit = time.time()

    copied_kwargs = MCMC_KWS.copy()
    mcmc_times = {}
    for steps in [1, 10]:
        start_mcmc = time.time()
        copied_kwargs.update({'steps': steps})
        _ = FITTER._mcmc(
            best_fit, data, mcmc_kws=copied_kwargs, npixels=NPIXELS)
        end_mcmc = time.time()
        mcmc_times.update({steps: end_mcmc - start_mcmc})
    print("Fitting time:\t{:.3} s".format(end_fit - start_fit))
    for steps, times in mcmc_times.items():
        print("{} MCMC steps:\t{:.3} s".format(steps, times))


def save_stuff(optimization_result, savefolder, which_image):
    mcmc_ml = optimization_result['mcmc_result']
    fit_ml = optimization_result['lmfit_result']
    best_ml = optimization_result['best_result']

    prefix = 'image-{}'.format(which_image)
    tosave = {'fit': fit_ml, 'mcmc': mcmc_ml, 'best': best_ml}
    for suffix, obj in tosave.items():
        filename = os.path.join(savefolder, prefix + '-{}.pkl'.format(suffix))
        inout.save_pickle(obj, filename)


if __name__ == '__main__':
    if WHICH_SPHERE == 'polystyrene':
        data_list, zpos = inout.load_polystyrene_sedimentation_data(
            size=SIZE, holonums=None)
        fits_mo, fits_ml = inout.load_polystyrene_sedimentation_params("03-27")
    elif WHICH_SPHERE == 'silica':
        data_list, zpos = inout.load_silica_sedimentation_data(
            size=SIZE, holonums=None)
        fits_mo, fits_ml = inout.load_silica_sedimentation_params("04-02")
    guesses = fits_mo if THEORY == 'mieonly' else fits_ml

    savefolder = '{}-{}-size={}-npx={}'.format(
        WHICH_SPHERE, THEORY, SIZE, NPIXELS)
    if not os.path.isdir(savefolder):
        os.mkdir(savefolder)

    time_mcmc(data_list[0], guesses['0'])
    raise ValueError
    for which_image, (data, guess) in enumerate(zip(data_list, guesses.values)):

        start_time = time.time()
        print("Starting {} mcmc, image {}".format(THEORY, which_image))
        optimization_result = FITTER.mcmc(
            guess, data, mcmc_kws=MCMC_KWS, npixels=NPIXELS)

        end_time = time.time()
        duration = timedelta(seconds=end_time - start_time)
        print("{} mcmc took {}".format(THEORY, duration))

        # Saving things...
        save_stuff(optimization_result, savefolder, which_image)

