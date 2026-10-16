import unittest

from src.nlp.grls_etl import build_normalized_row, extract_records, repair_mojibake
from src.nlp.grls_scraper import extract_records as scraper_extract_records


class GRLSExtractTests(unittest.TestCase):
    def test_top_level_records(self):
        payload = [{"lpId": "abc", "commercialName": "Drug", "atx": ["A01"]}]
        self.assertEqual(extract_records(payload), payload)
        self.assertEqual(scraper_extract_records(payload), payload)

    def test_envelope_records(self):
        payload = {"data": {"items": [{"lpId": "abc", "atx": ["J01"]}]}}
        expected = [{"lpId": "abc", "atx": ["J01"]}]
        self.assertEqual(extract_records(payload), expected)
        self.assertEqual(scraper_extract_records(payload), expected)

    def test_empty_page(self):
        self.assertEqual(extract_records([]), [])
        self.assertEqual(scraper_extract_records([]), [])

    def test_mojibake_repair(self):
        self.assertEqual(repair_mojibake("Р¦РµС„СѓСЂРѕРєСЃРёРј"), "Цефуроксим")
        self.assertEqual(repair_mojibake("Моксифлоксацин"), "Моксифлоксацин")

    def test_normalized_row_preserves_atx_and_raw_values(self):
        record = {
            "lpId": "abc",
            "commercialName": "Р—РёРЅРЅР°С‚В®",
            "mnnCode": ["3179"],
            "mnnName": "Р¦РµС„СѓСЂРѕРєСЃРёРј",
            "ruNumber": "П N015531/01",
            "atx": ["J01DC02"],
            "regdata": {"owner": "Компания", "entryDate": "2009-03-06"},
        }
        row = build_normalized_row(record, fix_mojibake=True)
        self.assertEqual(row["atx_codes"], ["J01DC02"])
        self.assertEqual(row["mnn_codes"], ["3179"])
        self.assertEqual(row["commercial_name"], "Зиннат®")
        self.assertEqual(row["commercial_name_raw"], "Р—РёРЅРЅР°С‚В®")
        self.assertEqual(row["mnn_name"], "Цефуроксим")


if __name__ == "__main__":
    unittest.main()
