import unittest

import numpy as np

from nilm_pipeline.extraction import extract_aligned_cycles, find_zero_crossings


class ExtractionTests(unittest.TestCase):
    def test_find_upward_zero_crossings(self):
        signal = np.asarray([-1, 1, 1, -1, -1, 1], dtype=float)
        self.assertEqual(find_zero_crossings(signal), [1, 5])

    def test_extract_fixed_length_cycles(self):
        phase = np.linspace(0, 8 * np.pi, 2000)
        voltage = np.sin(phase)
        current = 0.5 * np.cos(phase)
        cycles = extract_aligned_cycles(current, voltage, samples_per_cycle=250)
        self.assertGreater(len(cycles), 0)
        self.assertTrue(all(len(current_part) == 250 for current_part, _ in cycles))


if __name__ == "__main__":
    unittest.main()
