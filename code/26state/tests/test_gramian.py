import unittest

import numpy as np

from nilm_pipeline.gramian import angular_field, make_rgb_gramian


class GramianTests(unittest.TestCase):
    def test_field_shapes_and_ranges(self):
        signal = np.linspace(-1, 1, 32)
        for method in ("gasf", "gadf"):
            field = angular_field(signal, method=method)
            self.assertEqual(field.shape, (32, 32))
            self.assertTrue(np.isfinite(field).all())
            self.assertGreaterEqual(field.min(), 0)
            self.assertLessEqual(field.max(), 1)

    def test_constant_signal_is_finite(self):
        signal = np.ones(16)
        field = angular_field(signal, method="gasf")
        self.assertTrue(np.isfinite(field).all())

    def test_rgb_has_three_channels(self):
        signal = np.sin(np.linspace(0, 2 * np.pi, 20))
        rgb = make_rgb_gramian(signal, signal * 0.5, signal * 0.2)
        self.assertEqual(rgb.shape, (20, 20, 3))
        self.assertEqual(rgb.dtype, np.uint8)


if __name__ == "__main__":
    unittest.main()
