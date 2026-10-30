import unittest

import pandas as pd

from nilm_pipeline.splits import assert_no_group_overlap, assign_grouped_splits


class SplitTests(unittest.TestCase):
    def test_grouped_split_has_no_source_overlap(self):
        rows = []
        for label in ("A", "B"):
            for source in range(20):
                for cycle in range(5):
                    rows.append(
                        {
                            "label": label,
                            "source_id": f"{label}-source-{source}",
                            "image_path": f"{label}-{source}-{cycle}.png",
                        }
                    )
        frame = pd.DataFrame(rows)
        result = assign_grouped_splits(frame, seed=7)
        self.assertEqual(set(result["split"]), {"train", "val", "test"})
        assert_no_group_overlap(result)


if __name__ == "__main__":
    unittest.main()
