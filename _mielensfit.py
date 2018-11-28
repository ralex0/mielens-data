import numpy as np

import matplotlib.pyplot as plt

import holopy as hp
from holopy.scattering import calc_holo, Sphere
from holopy.scattering.theory import MieLens
from holopy.core.metadata import get_extents, get_spacing
from holopy.core.process import bg_correct, subimage, normalize, center_find
from holopy.inference import prior, AlphaModel, NmpfitStrategy
from holopy.inference.scipyfit import LeastSquaresScipyStrategy
from holopy.inference.model import PerfectLensModel

TESTHOLOINDEX = 1.49
TESTHOLORADIUS = .5
ZGUESS = 15.0

def _import_example_data():
    imagepath = hp.core.io.get_example_data_path('image01.jpg')
    raw_holo = hp.load_image(imagepath, spacing = 0.0851, medium_index = 1.33, 
                             illum_wavelen = 0.66, illum_polarization = (1,0))
    bgpath = hp.core.io.get_example_data_path(['bg01.jpg', 'bg02.jpg', 'bg03.jpg'])
    bg = hp.core.io.load_average(bgpath, refimg = raw_holo)
    holo = bg_correct(raw_holo, bg)
    holo = subimage(holo, [250,250], 200)
    holo = normalize(holo)
    return holo

def _fit_with_mielens_nmp(hologram):
    priors = _make_prior(hologram)
    model = PerfectLensModel(priors, noise_sd=hologram.noise_sd, lens_angle=prior.Uniform(0, 1.1, guess=0.8))
    result = NmpfitStrategy().fit(model, hologram)
    return result

def _fit_with_mieonly_nmp(hologram):
    priors = _make_prior(hologram)
    priors.center[2].lower_bound = 1e-16
    model = AlphaModel(priors, noise_sd=hologram.noise_sd, alpha=prior.Uniform(0, 1.0, guess=0.6))
    result = NmpfitStrategy().fit(model, hologram)
    return result

def _fit_with_mielens_scipy(hologram):
    priors = _make_prior(hologram)
    model = PerfectLensModel(priors, noise_sd=hologram.noise_sd, lens_angle=prior.Uniform(0, 1.1, guess=0.8))
    result = LeastSquaresScipyStrategy(random_subset=True).fit(model, hologram)
    return result

def _make_prior(hologram, index=TESTHOLOINDEX, radius=TESTHOLORADIUS):
    center = _make_center_priors_mielens(hologram)
    #center = prior.make_center_priors(hologram)
    #center[2] = prior.Uniform(-100, 100, guess = ZGUESS)
    s = Sphere(n=prior.Uniform(1.33, 1.65, guess=TESTHOLOINDEX), 
               r=prior.Uniform(0.5, 5, guess=TESTHOLORADIUS),
               center=center)
    return s

def _make_center_priors_mielens(im, zextent=5):
    extents = get_extents(im)
    extent = max(extents['x'], extents['y'])
    
    spacing = get_spacing(im)
    center = center_find(im) * spacing + [im.x[0], im.y[0]]

    xpar = prior.Uniform(im.x.values.min(), im.x.values.max(), guess=center[0])
    ypar = prior.Uniform(im.y.values.min(), im.y.values.max(), guess=center[1])
    zpar = prior.Uniform(-extent * zextent, extent * zextent, guess=ZGUESS)
    return xpar, ypar, zpar

def _calc_chisq_mielens(data, result):
    dt = data.values.squeeze()
    lens_angle = result.intervals[-1].guess
    fit = calc_holo(data, result.scatterer, theory=MieLens(lens_angle=lens_angle)).values.squeeze()
    return np.std(fit - dt) / np.std(dt)

def _calc_chisq_mieonly(data, result):
    dt = data.values.squeeze()
    scaling = result.intervals[-1].guess
    fit = calc_holo(data, result.scatterer, scaling=scaling).values.squeeze()
    return np.std(fit - dt) / np.std(dt)

def _calc_sum_res_sq_mielens(data, result):
    dt = data.values.squeeze()
    lens_angle = result.intervals[-1].guess
    fit = calc_holo(data, result.scatterer, theory=MieLens(lens_angle=lens_angle)).values.squeeze()
    return np.sum((fit - dt)**2)

def _calc_sum_res_sq_mieonly(data, result):
    dt = data.values.squeeze()
    scaling = result.intervals[-1].guess
    fit = calc_holo(data, result.scatterer, scaling=scaling).values.squeeze()
    return np.sum((fit - dt)**2)

def run_fit_plot_data(data):
    fit_mo = _fit_with_mieonly_nmp(data)
    chisq_mo = _calc_chisq_mieonly(data, fit_mo)
    fit_ml = _fit_with_mielens_nmp(data)
    chisq_ml = _calc_chisq_mielens(data, fit_ml)

    holo_dt = data.values.squeeze()
    holo_mo = calc_holo(data, fit_mo.scatterer, scaling=fit_mo.model.alpha).values.squeeze()
    holo_ml = calc_holo(data, fit_ml.scatterer, theory=MieLens(lens_angle=1.1)).values.squeeze()

    vmax = np.max((holo_ml, holo_mo, holo_dt))
    vmin = np.min((holo_ml, holo_mo, holo_dt))

    titles = ["Data", 
              "Best Fit Mie Only, chisq = {:.4f}".format(chisq_mo),
              "Best Fit mielens, chisq = {:.4f}".format(chisq_ml)]

    plot_three_things(holo_dt, holo_mo, holo_ml, titles)
    return fit_mo, fit_ml

def plot_three_things(data1, data2, data3, titles):
    vmax = np.max((data1, data2, data3))
    vmin = np.min((data1, data2, data3))

    plt.figure(figsize=(15,5))
    plt.gray()
    plt.subplot(131)
    plt.imshow(data1, interpolation="nearest", vmin=vmin, vmax=vmax)
    plt.colorbar()
    plt.title(titles[0])
    plt.subplot(132)
    plt.imshow(data2, interpolation="nearest", vmin=vmin, vmax=vmax)
    plt.colorbar()
    plt.title(titles[1])
    plt.subplot(133)
    plt.imshow(data3, interpolation="nearest", vmin=vmin, vmax=vmax)
    plt.colorbar()
    plt.title(titles[2])
    plt.show()