import unittest

import numpy as np

from nilm_11class.extraction import extract_aligned_cycles, find_zero_crossings


class ExtractionTests(unittest.TestCase):
    def test_crossings(self):
        signal = np.asarray([-1, 1, 1, -1, -1, 1], dtype=float)
        self.assertEqual(find_zero_crossings(signal), [1, 5])

    def test_cycles(self):
        phase = np.linspace(0, 8 * np.pi, 2000)
        cycles = extract_aligned_cycles(
            np.cos(phase),
            np.sin(phase),
            samples_per_cycle=250,
        )
        self.assertGreater(len(cycles), 0)
        self.assertTrue(all(len(cycle[0]) == 250 for cycle in cycles))


if __name__ == "__main__":
    unittest.main()
