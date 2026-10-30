import unittest

import pandas as pd

from nilm_11class.splits import assert_no_group_overlap, assign_grouped_splits


class SplitTests(unittest.TestCase):
    def test_no_group_overlap(self):
        rows = []
        for label in ("A", "B"):
            for source in range(20):
                for cycle in range(4):
                    rows.append(
                        {
                            "label": label,
                            "source_id": f"{label}-{source}",
                            "image_path": f"{label}-{source}-{cycle}.png",
                        }
                    )
        frame = assign_grouped_splits(pd.DataFrame(rows), seed=42)
        self.assertEqual(set(frame["split"]), {"train", "val", "test"})
        assert_no_group_overlap(frame)


if __name__ == "__main__":
    unittest.main()
