import unittest
from pathlib import Path

from nilm_pipeline.records import parse_26state_source, parse_state_from_name


class RecordTests(unittest.TestCase):
    def test_parse_state_from_name(self):
        self.assertEqual(
            parse_state_from_name("Air Conditioner_on_101441"),
            ("Air Conditioner", "on"),
        )
        self.assertEqual(parse_state_from_name("Fan_off"), ("Fan", "off"))

    def test_parse_26state_source(self):
        source = parse_26state_source(
            Path("Air Conditioner_on_101441/Air Conditioner_on_101441_535.pkl"),
            {
                "appliance_type": "Air Conditioner_on",
                "status": "101441",
                "file_id": "535",
                "regular_filename": "535_Air Conditioner_on_101441_116662.csv",
            },
            cycle_count=29,
        )
        self.assertEqual(source.label, "Air Conditioner_ON")
        self.assertTrue(source.source_id.endswith(".csv"))
        self.assertEqual(source.cycles, 29)


if __name__ == "__main__":
    unittest.main()
