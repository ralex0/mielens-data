import unittest

import holopy as hp

import sys
sys.path.append('../mielens-data')

from mielensfit import *

class TestIO(unittest.TestCase):
    def test_load_bgdivide_crop(self):
        data, guess_parameters = _import_example_data()
        self.assertTrue(True)


class TestFit(unittest.TestCase):
    def test_mielens_fits_with_nmp(self):
        data, guess_parameters = _import_example_data()
        fitresult = fit_mielens(data, guess_parameters, strategy='nmp')
        chisqB = calc_chisq_B(data, fitresult, theory='mielens')
        isok = chisqB < 1
        self.assertTrue(isok)

    def test_mielens_fits_with_scipy(self):
        data, guess_parameters = _import_example_data()
        fitresult = fit_mielens(data, guess_parameters, strategy='leastsquares')
        chisqB = calc_chisq_B(data, fitresult, theory='mielens')
        isok = chisqB < 1
        self.assertTrue(isok)

    unittest.skip("It's too slow")
    def test_mielens_inference(self):
        data, guess_parameters = _import_example_data()
        inference_result = mcmc_inference_mielens(data, guess_parameters)
        chisqB = calc_chisq_B(data, inference_result, theory='mielens')
        isok = chisqB < 1
        self.assertTrue(isok)

    def test_mieonly_fit(self):
        data, guess_parameters = _import_example_data()
        fitresult = fit_mieonly(data, guess_parameters)
        chisqB = calc_chisq_B(data, fitresult, theory='mieonly')
        isok = chisqB < 1
        self.assertTrue(isok)

    def test_mielens_globalop(self):
        data, guess_parameters = _import_example_data()
        fitresult = globalop_mielens(data, guess_parameters)
        chisqB = calc_chisq_B(data, fitresult, theory='mielens')
        isok = chisqB < 1
        self.assertTrue(isok)

    def test_mieonly_globalop(self):
        data, guess_parameters = _import_example_data()
        fitresult = globalop_mieonly(data, guess_parameters)
        chisqB = calc_chisq_B(data, fitresult, theory='mieonly')
        isok = chisqB < 1
        self.assertTrue(isok)


def _import_example_data():
    imagepath = hp.core.io.get_example_data_path('image01.jpg')
    metadata = {'spacing': 0.0851, 'medium_index': 1.33, 'illum_wavelen': 0.66, 
                'illum_polarization': (1,0)}
    particle_position = [250, 250]
    holo = load_bgdivide_crop(imagepath, metadata, particle_position,
                              bg_prefix="bg", size=200)
    guess_parameters = {'z': 15, 'r': .5, 'n': 1.58}
    return holo, guess_parameters

if __name__ == '__main__':
    unittest.main()