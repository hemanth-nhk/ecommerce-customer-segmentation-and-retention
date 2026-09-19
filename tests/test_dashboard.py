"""Automated test suite for Retention Engine Streamlit dashboard (app/dashboard.py)."""
import os
import unittest
from streamlit.testing.v1 import AppTest

DASHBOARD_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "app", "dashboard.py"))


class TestDashboard(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.at = AppTest.from_file(DASHBOARD_PATH, default_timeout=45).run()

    def test_app_loads_without_unhandled_exception(self):
        self.assertFalse(self.at.exception, f"App loaded with exception: {self.at.exception}")

    def test_navigation_buttons_present(self):
        # 6 navigation buttons should be present in sidebar
        btn_labels = [b.label for b in self.at.sidebar.button]
        expected_labels = [
            "Overview",
            "Customer Prediction",
            "Segmentation",
            "Model Performance",
            "Retention Analytics",
            "Methodology & Governance",
        ]
        self.assertEqual(btn_labels, expected_labels)

    def test_page_1_overview_renders(self):
        # Button 0 is Overview
        self.at.sidebar.button[0].click().run()
        self.assertFalse(self.at.exception, f"Overview threw exception: {self.at.exception}")
        markdown_text = " ".join([m.value for m in self.at.markdown])
        self.assertIn("Eligible Customers", markdown_text)
        self.assertIn("5,281", markdown_text)
        self.assertIn("43.40%", markdown_text)
        self.assertIn("£13.93M", markdown_text)
        # Verify defensive wording
        self.assertIn("showing discrimination and probability-error performance", markdown_text)
        self.assertIn("showing similar discrimination in the evaluated forward chronological validation", markdown_text)

    def test_page_2_customer_prediction_renders_and_infers(self):
        # Button 1 is Customer Prediction
        self.at.sidebar.button[1].click().run()
        self.assertFalse(self.at.exception, f"Customer Prediction threw exception: {self.at.exception}")
        metric_labels = [m.label for m in self.at.metric]
        self.assertIn("Estimated Probability of Future Commercial Purchase", metric_labels)
        prob_metric = next(m for m in self.at.metric if m.label == "Estimated Probability of Future Commercial Purchase")
        self.assertTrue(prob_metric.value.endswith("%"))
        prob_val = float(prob_metric.value.replace("%", ""))
        self.assertGreaterEqual(prob_val, 0.0)
        self.assertLessEqual(prob_val, 100.0)

    def test_page_3_segmentation_renders(self):
        # Button 2 is Segmentation
        self.at.sidebar.button[2].click().run()
        self.assertFalse(self.at.exception, f"Segmentation threw exception: {self.at.exception}")
        self.assertGreaterEqual(len(self.at.dataframe), 1, "Segmentation dataframe missing")
        df_seg = self.at.dataframe[0].value
        self.assertIn("RFM Segment", df_seg.columns)
        self.assertIn("Champions", list(df_seg["RFM Segment"]))

    def test_page_4_model_performance_renders(self):
        # Button 3 is Model Performance
        self.at.sidebar.button[3].click().run()
        self.assertFalse(self.at.exception, f"Model Performance threw exception: {self.at.exception}")

        # Check subheaders and dataframes
        subheader_texts = [sh.value for sh in self.at.subheader]
        self.assertTrue(any("Benchmark" in s for s in subheader_texts), "Benchmark subheader missing")
        self.assertTrue(any("Operating Point" in s for s in subheader_texts), "Operating point subheader missing")
        self.assertTrue(any("Forward Chronological" in s for s in subheader_texts), "Forward Chronological subheader missing")

        # Verify Benchmark DataFrame is present
        self.assertGreaterEqual(len(self.at.dataframe), 1, "M0-M5 benchmark table missing")
        df_bm = self.at.dataframe[0].value
        self.assertIn("Model", df_bm.columns)
        self.assertIn("PR-AUC", df_bm.columns)
        self.assertIn("ROC-AUC", df_bm.columns)
        self.assertIn("Brier Score", df_bm.columns)

        # Check model names in table
        models_in_table = list(df_bm["Model"])
        self.assertIn("M0 Majority Baseline", models_in_table)
        self.assertIn("M3 Logistic Regression", models_in_table)

        # Verify constant baseline top-decile lift is marked N/A
        m0_row = df_bm[df_bm["Model"] == "M0 Majority Baseline"].iloc[0]
        self.assertIn("N/A", str(m0_row["Top-Decile Lift"]))

        # Check candidate metrics at t* = 0.38
        metric_labels = [m.label for m in self.at.metric]
        self.assertIn("Tuned Threshold (t*)", metric_labels)
        self.assertIn("Held-Out Precision", metric_labels)
        self.assertIn("Held-Out Recall", metric_labels)
        self.assertIn("Held-Out F1-Score", metric_labels)

        metric_map = {m.label: m.value for m in self.at.metric}
        self.assertEqual(metric_map["Tuned Threshold (t*)"], "0.38")
        self.assertEqual(metric_map["Held-Out Precision"], "64.7%")
        self.assertEqual(metric_map["Held-Out Recall"], "77.1%")
        self.assertEqual(metric_map["Held-Out F1-Score"], "0.7038")

        # Check forward chronological metrics
        self.assertEqual(metric_map["Forward PR-AUC"], "0.7709")
        self.assertEqual(metric_map["Forward ROC-AUC"], "0.7994")
        self.assertEqual(metric_map["Forward Top-Decile Lift"], "2.1125x")
        self.assertEqual(metric_map["Forward Brier Score"], "0.2000")

    def test_page_5_retention_analytics_renders(self):
        # Button 4 is Retention Analytics
        self.at.sidebar.button[4].click().run()
        self.assertFalse(self.at.exception, f"Retention Analytics threw exception: {self.at.exception}")
        self.assertGreaterEqual(len(self.at.dataframe), 1, "Decile dataframe missing")
        df_dec = self.at.dataframe[0].value
        self.assertIn("Decile", df_dec.columns)
        self.assertEqual(len(df_dec), 10, "Should have 10 deciles")

    def test_page_6_methodology_and_governance_renders(self):
        # Button 5 is Methodology & Governance
        self.at.sidebar.button[5].click().run()
        self.assertFalse(self.at.exception, f"Methodology & Governance threw exception: {self.at.exception}")
        markdown_text = " ".join([m.value for m in self.at.markdown])
        self.assertIn("Online Retail II", markdown_text)
        self.assertIn("2011-09-10 12:50:00", markdown_text)
        self.assertIn("Non-Causal Association", markdown_text)


if __name__ == "__main__":
    unittest.main()
