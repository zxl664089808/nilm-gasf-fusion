import unittest

import numpy as np

from nilm_11class.gramian import angular_field, make_rgb_gramian


class GramianTests(unittest.TestCase):
    def test_shapes_and_ranges(self):
        signal = np.sin(np.linspace(0, 6.28, 40))
        for method in ("gasf", "gadf"):
            field = angular_field(signal, method=method)
            self.assertEqual(field.shape, (40, 40))
            self.assertTrue(np.isfinite(field).all())

    def test_rgb_shape(self):
        signal = np.sin(np.linspace(0, 6.28, 20))
        rgb = make_rgb_gramian(signal, signal * 0.5, signal * 0.1)
        self.assertEqual(rgb.shape, (20, 20, 3))
        self.assertEqual(rgb.dtype, np.uint8)


if __name__ == "__main__":
    unittest.main()
