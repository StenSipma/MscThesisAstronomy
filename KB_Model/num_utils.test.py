import unittest

import num_utils
import numba
import numpy as np


class TestUtil(unittest.TestCase):
    def test_find_root_secant(self):
        @numba.jit(nopython=True)
        def test_func(x, _args):
            return x - 1

        found_x = num_utils.find_root_secant(test_func, 10, -10, tol=1e-6)
        self.assertTrue(abs(found_x - 1) <= 1e-6)

        @numba.jit(nopython=True)
        def test_func_2(x, _args):
            return x + 1

        found_x = num_utils.find_root_secant(test_func_2, 2e-3 - 1, -1e-3 - 1, tol=1e-6)
        self.assertTrue(abs(found_x + 1) <= 1e-6)

    def test_find_segment(self):
        test_list = np.arange(10)

        self.assertEqual(num_utils.find_segment(5.5, test_list), 5)
        self.assertEqual(num_utils.find_segment(0, test_list), 0)
        self.assertEqual(num_utils.find_segment(5, test_list), 5)

        # TODO: fix this test case ()
        # assert num_utils.find_segment(9, test_list) == 8

        # TODO: implement extrapolation (check if it works with interp.)
        self.assertEqual(num_utils.find_segment(-10, test_list), 0)
        self.assertEqual(num_utils.find_segment(20, test_list), 8)

    def test_linear_spline(self):
        x_test_list = np.arange(10)
        y_test_list = x_test_list + 1

        self.assertEqual(num_utils.linear_spline(0, x_test_list, y_test_list), 1)
        self.assertEqual(num_utils.linear_spline(7, x_test_list, y_test_list), 8)
        self.assertEqual(num_utils.linear_spline(3.1234, x_test_list, y_test_list), 4.1234)
        self.assertEqual(num_utils.linear_spline(9, x_test_list, y_test_list), 10)

        # Test extrapolation ?
        self.assertEqual(num_utils.linear_spline(20, x_test_list, y_test_list), 21)
        self.assertEqual(num_utils.linear_spline(-30, x_test_list, y_test_list), -29)

        # TODO: add more test cases for nonlinear functions

    def test_bisect(self):
        @numba.jit(nopython=True)
        def test_func(x):
            return x - 1

        found_x = num_utils.bisect(test_func, 10, -10, x_tol=1e-10)
        self.assertTrue(abs(found_x - 1) <= 1e-10)

        # Works both ways:
        found_x = num_utils.bisect(test_func, -10, 10, x_tol=1e-10)
        self.assertTrue(abs(found_x - 1) <= 1e-10)

        @numba.jit(nopython=True)
        def test_func_2(x):
            return x

        found_x = num_utils.bisect(test_func_2, 1e-10, -1e-10, x_tol=1e-20)
        self.assertTrue(abs(found_x) <= 1e-20)

        try:
            found_x = num_utils.bisect(test_func, -10, -8, x_tol=1e-10)
            self.fail("This call must raise a ValueError")
        except ValueError:
            pass
        except Exception:
            self.fail("Call raised the wrong exception")

        try:
            found_x = num_utils.bisect(test_func, -10, 10, x_tol=1e-50, max_iter=5)
            self.fail("This call must raise a RuntimeError")
        except RuntimeError:
            pass
        except Exception:
            self.fail("Call raised the wrong exception")


if __name__ == "__main__":
    unittest.main()
