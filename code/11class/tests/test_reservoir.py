import tempfile
import unittest
from pathlib import Path

import numpy as np

from importlib.util import module_from_spec, spec_from_file_location


class ReservoirTests(unittest.TestCase):
    def test_reservoir_returns_fixed_size(self):
        script = Path(__file__).parents[1] / "scripts" / "generate_dataset.py"
        spec = spec_from_file_location("generate_dataset", script)
        module = module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(module)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            phase = np.linspace(0, 50 * np.pi, 12000)
            data = np.column_stack([np.cos(phase), np.sin(phase)])
            np.savetxt(root / "sample.csv", data, delimiter=",")
            samples = [{"csv_file": "sample.csv", "original_index": 1}]
            selected = module.reservoir_sample(
                samples,
                root,
                maximum=20,
                samples_per_cycle=500,
                seed=42,
                label="Test",
            )
            self.assertEqual(len(selected), 20)
            self.assertTrue(all(item["source_id"] for item in selected))


if __name__ == "__main__":
    unittest.main()
