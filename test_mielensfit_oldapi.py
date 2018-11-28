import unittest

import numpy as np

import matplotlib.pyplot as plt

import holopy as hp
from holopy.core.process import bg_correct, subimage, normalize, center_find
from holopy.core.metadata import get_extents, get_spacing
from holopy.fitting import fit, Model
from holopy.fitting import Parameter
from holopy.scattering import calc_holo, Sphere
from holopy.scattering.theory import MieLens

TESTHOLOINDEX = 1.58
TESTHOLORADIUS = .5
ZGUESS = 15

class TestFit(unittest.TestCase):
    def test_fit_mieonly(self):
        data_hologram = _import_example_data()
        fit_result = _fit_mieonly(data_hologram)
        chisq = _calc_chisq_mieonly(data_hologram, fit_result)
        isok = chisq < 1
        if not isok:
            print(chisq)
        self.assertTrue(isok)

    def test_fit_mielens(self):
        data_hologram = _import_example_data()
        fit_result = _fit_mielens(data_hologram)
        chisq = _calc_chisq_mielens(data_hologram, fit_result)
        isok = chisq < 1
        if not isok:
            print(chisq)
        self.assertTrue(isok)

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

def _fit_mieonly(hologram):
    parameters = _make_parameters_mieonly(hologram)
    model = Model(parameters, calc_holo, alpha=Parameter(.6, [.1, 1]))
    result = fit(model, hologram)
    return result

def _fit_mielens(hologram):
    parameters = _make_parameters_mielens(hologram)
    model = Model(parameters, calc_holo, theory=MieLens(lens_angle=1.1))
    result = fit(model, hologram)
    return result

def _make_parameters_mieonly(hologram, zextent=5):
    center = _make_center_parameter_mieonly(hologram, zextent)
    r = Parameter(TESTHOLORADIUS, [0.005, 5])
    n = Parameter(TESTHOLOINDEX, [1.33, 1.65])
    return Sphere(center=center, r=r, n=n)

def _make_parameters_mielens(hologram, zextent=5):
    center = _make_center_parameter_mielens(hologram, zextent)
    r = Parameter(TESTHOLORADIUS, [0.005, 5])
    n = Parameter(TESTHOLOINDEX, [1.33, 1.65])
    return Sphere(center=center, r=r, n=n)

def _make_center_parameter_mieonly(im, zextent=5):
    extents = get_extents(im)
    extent = max(extents['x'], extents['y'])
    zrange = [0, zextent*extent]
    
    spacing = get_spacing(im)
    center = center_find(im) * spacing + [im.x[0], im.y[0]]

    xpar = Parameter(center[0], [im.x.values.min(), im.x.values.max()])
    ypar = Parameter(center[1], [im.y.values.min(), im.y.values.max()])
    zpar = Parameter(ZGUESS, zrange)
    return xpar, ypar, zpar

def _make_center_parameter_mielens(im, zextent=5):
    xpar, ypar, zpar = _make_center_parameter_mieonly(im, zextent=zextent)
    return xpar, ypar, Parameter(ZGUESS, [-max(zpar.limit), max(zpar.limit)])

def _calc_chisq_mielens(data, result):
    dt = data.values.squeeze()
    fit = calc_holo(data, result.scatterer, theory=MieLens(lens_angle=1.1)).values.squeeze()
    return np.std(fit - dt) / np.std(dt)

def _calc_chisq_mieonly(data, result):
    dt = data.values.squeeze()
    fit = calc_holo(data, result.scatterer, scaling=result.alpha).values.squeeze()
    return np.std(fit - dt) / np.std(dt)

def run_fit_plot_example_data():
    data = _import_example_data()
    
    fit_mo = _fit_mieonly(data)
    chisq_mo = _calc_chisq_mieonly(data, fit_mo)
    fit_ml = _fit_mielens(data)
    chisq_ml = _calc_chisq_mielens(data, fit_ml)

    holo_dt = data.values.squeeze()
    holo_mo = calc_holo(data, fit_mo.scatterer, scaling=fit_mo.alpha).values.squeeze()
    holo_ml = calc_holo(data, fit_ml.scatterer, theory=MieLens(lens_angle=1.1)).values.squeeze()

    vmax = np.max((holo_ml, holo_mo, holo_dt))
    vmin = np.min((holo_ml, holo_mo, holo_dt))

    
    plt.figure(figsize=(15,5))
    plt.subplot(131)
    plt.imshow(holo_dt, interpolation="nearest", vmin=vmin, vmax=vmax)
    plt.colorbar()
    plt.title("Example Data")
    plt.subplot(132)
    plt.imshow(holo_mo, interpolation="nearest", vmin=vmin, vmax=vmax)
    plt.colorbar()
    plt.title("Mie Only, chisq = {}".format(chisq_mo))
    plt.subplot(133)
    plt.imshow(holo_ml, interpolation="nearest", vmin=vmin, vmax=vmax)
    plt.colorbar()
    plt.title("mielens, chisq = {}".format(chisq_ml))
    plt.gray()
    plt.show()

if __name__ == '__main__':
    unittest.main()
#    run_fit_plot_example_data()