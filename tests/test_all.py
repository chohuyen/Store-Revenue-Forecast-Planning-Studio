"""Kiểm thử: dữ liệu, pipeline, mô hình, What-if, ứng dụng web. Chạy: python -m unittest discover -s tests -v"""
import unittest
import numpy as np
import pandas as pd
from src import config
from src.data.loader import find_data_dir, load_train_store
from src.features.pipeline import RossmannDataPipeline
from src.models.evaluate import rmspe
from src.models.predict import Forecaster, InputError, validate_inputs
from src.models.whatif import compare, run_what_if, four_scenarios

HAS_MODEL = config.BUNDLE_PATH.exists()
try:
    find_data_dir(); HAS_DATA = True
except FileNotFoundError:
    HAS_DATA = False


@unittest.skipUnless(HAS_DATA, "thiếu dữ liệu")
class TestData(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tr, cls.st = load_train_store()

    def test_shapes(self):
        self.assertEqual(self.tr.shape, (1017209, 9))
        self.assertEqual(self.st.shape, (1115, 10))

    def test_no_duplicates(self):
        self.assertEqual(int(self.tr.duplicated().sum()), 0)

    def test_state_holiday_codes(self):
        self.assertEqual(set(self.tr["StateHoliday"].astype(str)), {"0", "a", "b", "c"})

    def test_pipeline_one_command(self):
        out = RossmannDataPipeline().fit(self.st).transform(self.tr, self.st)
        self.assertEqual(out.shape, (844338, 30))
        for c in config.FEATURES:
            self.assertIn(c, out.columns)
        self.assertNotIn("Customers", config.FEATURES)
        self.assertFalse(out[config.FEATURES].isnull().any().any())


class TestMetrics(unittest.TestCase):
    def test_rmspe_ignores_zero_actual(self):
        self.assertAlmostEqual(rmspe([100, 0], [110, 5]), 0.1)

    def test_rmspe_all_zero(self):
        self.assertTrue(np.isnan(rmspe([0, 0], [1, 1])))

    def test_compare(self):
        d, p = compare(1000, 1100); self.assertAlmostEqual(d, 100); self.assertAlmostEqual(p, 10.0)
        d, p = compare(0, 50); self.assertEqual(d, 50); self.assertIsNone(p)


class TestValidation(unittest.TestCase):
    def test_bad_inputs(self):
        for args in [("abc", "2015-08-15", 0, "0", 0), (1, "khong-phai-ngay", 0, "0", 0),
                     (1, "1999-01-01", 0, "0", 0), (1, "2015-08-15", 2, "0", 0),
                     (1, "2015-08-15", 0, "x", 0), (1, "2015-08-15", 0, "0", 5)]:
            with self.assertRaises(InputError):
                validate_inputs(*args)

    def test_unknown_store(self):
        with self.assertRaises(InputError):
            validate_inputs(5000, "2015-08-15", 0, "0", 0, valid_stores=set(range(1, 1116)))


@unittest.skipUnless(HAS_MODEL and HAS_DATA, "thiếu mô hình/dữ liệu")
class TestModelAndWhatIf(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.fc = Forecaster()

    def sc(self, store=1, date="2015-08-15", promo=0, h="0", school=0):
        return self.fc.validate(store, date, promo, h, school)

    def test_reload_and_reproducible(self):
        a, _ = self.fc.predict_one(self.sc()); b, _ = Forecaster().predict_one(self.sc())
        self.assertEqual(a, b); self.assertGreater(a, 0)

    def test_feature_order_matches_training(self):
        _, row = self.fc.predict_one(self.sc())
        self.assertTrue(set(self.fc.bundle.features) <= set(row.index))
        self.assertEqual(self.fc.bundle.features, config.FEATURES)

    def test_promo_on_off_only_changes_promo_features(self):
        base = self.sc(promo=0)
        _, f = self.fc.predict_scenarios([base, self.sc(promo=1)])
        diff = [c for c in config.FEATURES if f.iloc[0][c] != f.iloc[1][c]]
        self.assertTrue(set(diff) <= {"Promo", "Promo_x_SchoolHoliday", "Promo_x_StateHoliday"})
        self.assertIn("Promo", diff)

    def test_whatif_math(self):
        r = run_what_if(self.fc, self.sc(promo=0), promo=1)
        self.assertAlmostEqual(r["diff"], r["new_sales"] - r["base_sales"])
        self.assertAlmostEqual(r["pct"], r["diff"] / r["base_sales"] * 100)

    def test_same_scenario_zero_diff(self):
        r = run_what_if(self.fc, self.sc(promo=1), promo=1)
        self.assertEqual(r["diff"], 0.0); self.assertEqual(r["pct"], 0.0)

    def test_holiday_change(self):
        """Mô hình có phản ứng với cờ ngày lễ ở phần lớn tổ hợp (không bắt buộc ở từng ca)."""
        changed = 0
        for s in (1, 50, 262, 500, 817, 1000):
            for d in ("2014-03-10", "2015-07-20"):
                r = run_what_if(self.fc, self.sc(s, d), state_holiday="a")
                self.assertTrue(np.isfinite(r["new_sales"]))
                changed += r["diff"] != 0
        self.assertGreater(changed, 0)

    def test_holiday_types_flag(self):
        r = run_what_if(self.fc, self.sc(h="a"), state_holiday="b")
        if not self.fc.bundle.supports_holiday_type:
            self.assertEqual(r["diff"], 0.0); self.assertTrue(r["notes"])

    def test_combined_and_four(self):
        r = run_what_if(self.fc, self.sc(), promo=1, state_holiday="a"); self.assertTrue(np.isfinite(r["new_sales"]))
        rows = four_scenarios(self.fc, self.sc(), "a")
        self.assertEqual(len(rows), 4); self.assertEqual(rows[0]["diff_vs_kb1"], 0.0)

    def test_multiple_stores_and_dates(self):
        vals = {self.fc.predict_one(self.sc(s, d))[0] for s in (1, 85, 262, 1115) for d in ("2014-03-10", "2015-07-04", "2015-09-01")}
        self.assertGreater(len(vals), 6)


@unittest.skipUnless(HAS_MODEL and HAS_DATA, "thiếu mô hình/dữ liệu")
class TestWeb(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from app.app import app
        cls.c = app.test_client()

    def test_pages(self):
        for p in ("/", "/forecast", "/whatif", "/analysis", "/health"):
            self.assertEqual(self.c.get(p).status_code, 200, p)

    def test_forecast_post(self):
        r = self.c.post("/forecast", data={"store": "1", "date": "2015-08-15", "promo": "1", "state_holiday": "0", "school_holiday": "0"})
        self.assertIn("Doanh thu dự báo", r.get_data(as_text=True))

    def test_forecast_bad_input_no_crash(self):
        r = self.c.post("/forecast", data={"store": "0", "date": "2015-08-15", "promo": "1", "state_holiday": "0", "school_holiday": "0"})
        self.assertEqual(r.status_code, 200); self.assertIn("không tồn tại", r.get_data(as_text=True))

    def test_whatif_flow(self):
        r = self.c.post("/whatif", data={"store": "1", "date": "2015-07-20", "action": "load"})
        self.assertIn("Kịch bản ban đầu", r.get_data(as_text=True))
        r = self.c.post("/whatif", data={"store": "1", "date": "2015-07-20", "promo": "0", "state_holiday": "0", "school_holiday": "0",
                                         "new_promo": "1", "new_holiday": "0", "action": "run"})
        t = r.get_data(as_text=True); self.assertIn("Chênh lệch doanh thu", t); self.assertIn("So sánh 4 kịch bản", t)

    def test_api(self):
        r = self.c.post("/api/predict", json={"store": 1, "date": "2015-08-15", "promo": 1, "state_holiday": "0", "school_holiday": 0})
        self.assertEqual(r.status_code, 200); self.assertGreater(r.get_json()["predicted_sales"], 0)
        r = self.c.post("/api/whatif", json={"store": 1, "date": "2015-08-15", "promo": 0, "new_promo": 1})
        self.assertEqual(r.status_code, 200); self.assertIn("pct", r.get_json())
        self.assertEqual(self.c.post("/api/predict", json={"store": 99999, "date": "2015-08-15"}).status_code, 400)


if __name__ == "__main__":
    unittest.main()
