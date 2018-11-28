import unittest

import sys
sys.path.append('../mielens-data')

import _mielensfit as mlf

class TestFit(unittest.TestCase):
    def test_mielens_fits_with_nmp(self):
        data_hologram = mlf._import_example_data()
        fit_result = mlf._fit_with_mielens_nmp(data_hologram)
        chisq = mlf._calc_chisq_mielens(data_hologram, fit_result)
        isok = chisq < 1
        if not isok:
            print(chisq)
        self.assertTrue(isok)

    def test_mielens_fits_with_scipy(self):
        data_hologram = mlf._import_example_data()
        fit_result = mlf._fit_with_mielens_scipy(data_hologram)
        chisq = mlf._calc_chisq_mielens(data_hologram, fit_result)
        isok = chisq < 1
        if not isok:
            print(chisq)
        self.assertTrue(isok)

if __name__ == '__main__':
    unittest.main()