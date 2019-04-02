import unittest

import numpy as np

import holopy as hp

import sys
sys.path.append('../mielens-data')

from mielensfit import *

class TestIO(unittest.TestCase):
    def test_load_bgdivide_crop(self):
        data, guess_parameters = _import_example_data()
        self.assertTrue(True)


class TestFit(unittest.TestCase):
    def test_mielens_fit(self):
        raise NotImplementedError("test is now outdated")

    unittest.skip("It's too slow")
    def test_mielens_inference(self):
        raise NotImplementedError("test is now outdated")

    def test_mieonly_fit(self):
        raise NotImplementedError("test is now outdated")


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
