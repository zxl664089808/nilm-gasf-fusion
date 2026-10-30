import tempfile
import unittest
from pathlib import Path

import numpy as np

from nilm_11class.image_io import save_pure_rgb, title_band_report


class ImageTests(unittest.TestCase):
    def test_pure_image(self):
        rgb = np.zeros((64, 64, 3), dtype=np.uint8)
        rgb[..., 1] = 170
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "image.png"
            save_pure_rgb(path, rgb)
            self.assertFalse(title_band_report(path)["suspicious_title_band"])


if __name__ == "__main__":
    unittest.main()
