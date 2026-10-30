import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from nilm_pipeline.image_io import save_pure_rgb, title_band_report


class ImageIOTests(unittest.TestCase):
    def test_save_pure_rgb_has_no_white_title_band(self):
        rgb = np.zeros((64, 64, 3), dtype=np.uint8)
        rgb[..., 0] = 180
        rgb[..., 1] = 20
        rgb[..., 2] = 40
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "sample.png"
            save_pure_rgb(path, rgb)
            loaded = np.asarray(Image.open(path).convert("RGB"))
            self.assertEqual(loaded.shape, (64, 64, 3))
            self.assertTrue(np.array_equal(loaded, rgb))
            self.assertFalse(title_band_report(path)["suspicious_title_band"])


if __name__ == "__main__":
    unittest.main()
