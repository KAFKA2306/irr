import unittest

from src.estat_contract import validate_contract


class EstatContractTests(unittest.TestCase):
    def test_quarantine_file_cannot_contain_executable_queries(self):
        config = {
            "schema_version": 1,
            "verified": False,
            "quarantine_reason": "metadata not verified",
            "queries": [{"industry": "産業計"}],
        }
        errors = validate_contract(config, require_verified=False)
        self.assertTrue(any("must not contain executable queries" in item for item in errors))

    def test_verified_contract_rejects_wildcard_selectors(self):
        spec = {
            "stats_data_id": "123456",
            "table_title": "Official table",
            "source_url": "https://www.e-stat.go.jp/example",
            "retrieved_at": "2026-08-02",
            "metadata_sha256": "0" * 64,
            "population_scope": "一般労働者",
            "company_size_label": "1000人以上",
            "sex_label": "男女計",
            "age_label": "年齢計",
            "measure_label": "きまって支給する現金給与額",
            "expected_unit": "千円",
            "frequency": "annual",
            "filters": {"cdCat01": "*"},
            "expected_dimensions": {"産業": "産業計"},
        }
        config = {
            "schema_version": 1,
            "verified": True,
            "queries": [{"industry": "産業計", "wage": spec, "bonus": spec}],
        }
        errors = validate_contract(config)
        self.assertTrue(any("wildcard" in item for item in errors))

    def test_complete_verified_contract_is_accepted(self):
        base = {
            "stats_data_id": "123456",
            "table_title": "Official table",
            "source_url": "https://www.e-stat.go.jp/example",
            "retrieved_at": "2026-08-02",
            "metadata_sha256": "a" * 64,
            "population_scope": "一般労働者",
            "company_size_label": "1000人以上",
            "sex_label": "男女計",
            "age_label": "年齢計",
            "frequency": "annual",
            "filters": {"cdCat01": "01", "cdCat02": "02"},
            "expected_dimensions": {"産業": "産業計", "企業規模": "1000人以上"},
        }
        wage = {**base, "measure_label": "月額賃金", "expected_unit": "千円"}
        bonus = {**base, "measure_label": "年間賞与", "expected_unit": "千円"}
        config = {
            "schema_version": 1,
            "verified": True,
            "queries": [{"industry": "産業計", "wage": wage, "bonus": bonus}],
        }
        self.assertEqual(validate_contract(config), [])


if __name__ == "__main__":
    unittest.main()
