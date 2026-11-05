#!/usr/bin/env python3
"""
Unit and functional test suite for Isolation Forest Network Anomaly Detector.
"""

import math
import os
import tempfile
import time
import unittest

import numpy as np
import pandas as pd

from anomaly_detector import (
    AnomalyResult,
    CyclicalTimeEncoder,
    HourlyBaseline,
    NetworkAnomalyDetector,
    generate_synthetic_diurnal_dataset,
)


class TestCyclicalTimeEncoder(unittest.TestCase):
    def test_trigonometric_anchors(self):
        # Base midnight timestamp: 1700000000 // 86400 * 86400
        base_midnight = float((1700000000 // 86400) * 86400)

        # 00:00 Midnight: sin(0) = 0, cos(0) = 1
        h0, sin0, cos0 = CyclicalTimeEncoder.encode_timestamp(base_midnight)
        self.assertAlmostEqual(h0, 0.0, places=4)
        self.assertAlmostEqual(sin0, 0.0, places=4)
        self.assertAlmostEqual(cos0, 1.0, places=4)

        # 06:00 AM: sin(pi/2) = 1, cos(pi/2) = 0
        h6, sin6, cos6 = CyclicalTimeEncoder.encode_timestamp(base_midnight + 6 * 3600)
        self.assertAlmostEqual(h6, 6.0, places=4)
        self.assertAlmostEqual(sin6, 1.0, places=4)
        self.assertAlmostEqual(cos6, 0.0, places=4)

        # 12:00 Noon: sin(pi) = 0, cos(pi) = -1
        h12, sin12, cos12 = CyclicalTimeEncoder.encode_timestamp(base_midnight + 12 * 3600)
        self.assertAlmostEqual(h12, 12.0, places=4)
        self.assertAlmostEqual(sin12, 0.0, places=4)
        self.assertAlmostEqual(cos12, -1.0, places=4)

        # 18:00 PM: sin(3pi/2) = -1, cos(3pi/2) = 0
        h18, sin18, cos18 = CyclicalTimeEncoder.encode_timestamp(base_midnight + 18 * 3600)
        self.assertAlmostEqual(h18, 18.0, places=4)
        self.assertAlmostEqual(sin18, -1.0, places=4)
        self.assertAlmostEqual(cos18, 0.0, places=4)

    def test_smooth_continuity_across_midnight(self):
        base_midnight = float((1700000000 // 86400) * 86400)
        ts_2359 = base_midnight - 60.0  # 23:59 previous day
        ts_0001 = base_midnight + 60.0  # 00:01 current day

        _, sin_2359, cos_2359 = CyclicalTimeEncoder.encode_timestamp(ts_2359)
        _, sin_0001, cos_0001 = CyclicalTimeEncoder.encode_timestamp(ts_0001)

        # Euclidean distance in 2D cyclical space should be very small (~ 2 minutes apart)
        dist = math.sqrt((sin_2359 - sin_0001) ** 2 + (cos_2359 - cos_0001) ** 2)
        self.assertLess(dist, 0.05, "Distance across midnight must be smoothly continuous")

    def test_vectorized_series_encoding(self):
        timestamps = pd.Series([1700000000.0, 1700003600.0, 1700007200.0])
        sin_vals, cos_vals = CyclicalTimeEncoder.encode_series(timestamps)

        self.assertEqual(len(sin_vals), 3)
        self.assertEqual(len(cos_vals), 3)

        # Verify unit circle radius: sin^2 + cos^2 = 1.0
        for s, c in zip(sin_vals, cos_vals):
            self.assertAlmostEqual(s ** 2 + c ** 2, 1.0, places=5)


class TestHourlyVolumeBaseline(unittest.TestCase):
    def setUp(self):
        base_midnight = float((1700000000 // 86400) * 86400)
        records = []

        # Create 2 days of synthetic data where hour 14 has high volume and hour 3 has low volume
        for day in range(2):
            for h in range(24):
                ts = base_midnight + (day * 86400) + (h * 3600)
                # Hour 14: mean 5000, std ~ 100
                # Hour 3: mean 500, std ~ 50
                mean_vol = 5000.0 if h == 14 else (500.0 if h == 3 else 2000.0)
                for i in range(5):
                    vol = mean_vol + (i - 2) * 200.0
                    records.append({
                        "timestamp": ts + (i * 60),
                        "bytes_per_sec": vol,
                        "packets_per_sec": vol / 100.0,
                        "byte_count": vol * 5,
                    })

        self.df = pd.DataFrame(records)
        self.baseline = HourlyBaseline()
        self.baseline.fit(self.df)

    def test_hourly_zscore_differentiation(self):
        # 1. High business hour traffic (5200 B/s at 14:00) -> should be close to 0 sigma
        z_business = self.baseline.compute_zscore("bytes_per_sec", 5200.0, hour=14)
        self.assertLess(abs(z_business), 3.0, "5200 B/s at 14:00 must yield normal Z-score")

        # 2. Same 5200 B/s traffic occurring at 03:00 (quiet night hour where mean is 500)
        z_night = self.baseline.compute_zscore("bytes_per_sec", 5200.0, hour=3)
        self.assertGreater(z_night, 15.0, "5200 B/s at 03:00 must yield massive Z-score (> 15 sigma)")

    def test_zero_variance_safety(self):
        df_const = pd.DataFrame({
            "timestamp": [1700000000.0, 1700000010.0],
            "bytes_per_sec": [100.0, 100.0],  # Variance = 0
            "packets_per_sec": [1.0, 1.0],
            "byte_count": [100.0, 100.0],
        })
        b = HourlyBaseline()
        b.fit(df_const)
        # Should not raise ZeroDivisionError
        z = b.compute_zscore("bytes_per_sec", 100.0, hour=0)
        self.assertAlmostEqual(z, 0.0)


class TestNetworkAnomalyDetector(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Train on a 7-day diurnal synthetic dataset
        cls.train_df = generate_synthetic_diurnal_dataset(num_days=7, samples_per_hour=30, seed=42)
        cls.detector = NetworkAnomalyDetector(n_estimators=100, contamination=0.02, random_state=42)
        cls.detector.fit(cls.train_df)

    def test_feature_extraction_completeness(self):
        sample_flow = {
            "timestamp": 1700000000.0,
            "bytes_per_sec": 1500.0,
            "packets_per_sec": 12.0,
            "packet_count": 24,
            "byte_count": 3000,
            "flow_duration": 2.0,
            "syn_flag_ratio": 0.04,
            "tcp_rst_ratio": 0.0,
        }
        vec, features_dict, zscores = self.detector.extract_features_single(sample_flow)

        self.assertEqual(vec.shape, (1, len(self.detector.FEATURE_COLUMNS)))
        for col in self.detector.FEATURE_COLUMNS:
            self.assertIn(col, features_dict)
        self.assertIn("hour_sin", features_dict)
        self.assertIn("hour_cos", features_dict)
        self.assertIn("zscore_bytes_per_sec", zscores)

    def test_business_hour_traffic_not_flagged_as_anomaly(self):
        # 14:00 UTC (peak business hours) high legitimate volume
        base_midnight = float((1700000000 // 86400) * 86400)
        ts_1400 = base_midnight + (8 * 86400.0) + (14 * 3600.0)

        busy_business_flow = {
            "timestamp": ts_1400,
            "src_ip": "192.168.1.100",
            "src_port": 50000,
            "dst_ip": "142.250.190.46",
            "dst_port": 443,
            "protocol": "TCP",
            "flow_duration": 4.0,
            "bytes_per_sec": 5000.0,  # High volume typical of 14:00
            "packets_per_sec": 40.0,
            "packet_count": 160,
            "byte_count": 20000,
            "syn_flag_ratio": 0.01,
            "tcp_rst_ratio": 0.0,
        }

        res = self.detector.predict_flow(busy_business_flow)
        self.assertFalse(res.is_anomaly, "Legitimate business-hour volume must not be flagged as anomaly")
        self.assertEqual(res.severity, "NORMAL")
        self.assertLess(res.anomaly_score, 0.50)

    def test_off_hours_exfiltration_flagged_as_critical_anomaly(self):
        # 03:00 UTC (quiet off-hours) massive data burst
        base_midnight = float((1700000000 // 86400) * 86400)
        ts_0300 = base_midnight + (8 * 86400.0) + (3 * 3600.0)

        exfiltration_flow = {
            "timestamp": ts_0300,
            "src_ip": "192.168.1.200",
            "src_port": 39999,
            "dst_ip": "198.51.100.88",
            "dst_port": 8443,
            "protocol": "TCP",
            "flow_duration": 3.0,
            "bytes_per_sec": 15000.0,  # Abnormal volume at 03:00!
            "packets_per_sec": 100.0,
            "packet_count": 300,
            "byte_count": 45000,
            "syn_flag_ratio": 0.0,
            "tcp_rst_ratio": 0.0,
        }

        res = self.detector.predict_flow(exfiltration_flow)
        self.assertTrue(res.is_anomaly, "Off-hours data burst must be detected as anomaly")
        self.assertEqual(res.severity, "CRITICAL")
        self.assertGreater(res.anomaly_score, 0.70)
        self.assertGreater(res.zscores["zscore_bytes_per_sec"], 10.0)

    def test_abnormal_syn_scan_flagged(self):
        # Abnormal SYN flood/scan profile
        base_midnight = float((1700000000 // 86400) * 86400)
        ts_1400 = base_midnight + (8 * 86400.0) + (14 * 3600.0)

        syn_scan_flow = {
            "timestamp": ts_1400,
            "src_ip": "10.0.0.1",
            "src_port": 40001,
            "dst_ip": "192.168.1.50",
            "dst_port": 22,
            "protocol": "TCP",
            "flow_duration": 0.05,
            "bytes_per_sec": 1200.0,
            "packets_per_sec": 20.0,
            "packet_count": 1,
            "byte_count": 60,
            "syn_flag_ratio": 1.0,  # 100% SYN
            "tcp_rst_ratio": 0.0,
        }

        res = self.detector.predict_flow(syn_scan_flow)
        self.assertTrue(res.is_anomaly, "SYN scan profile must be flagged as anomaly")

    def test_batch_prediction(self):
        base_midnight = float((1700000000 // 86400) * 86400)
        ts_1400 = base_midnight + (8 * 86400.0) + (14 * 3600.0)

        flows = [
            {
                "timestamp": ts_1400,
                "bytes_per_sec": 5000.0,
                "packets_per_sec": 40.0,
                "packet_count": 160,
                "byte_count": 20000,
                "flow_duration": 4.0,
                "syn_flag_ratio": 0.01,
                "tcp_rst_ratio": 0.0,
            },
            {
                "timestamp": ts_1400,
                "bytes_per_sec": 20000.0,
                "packets_per_sec": 150.0,
                "packet_count": 600,
                "byte_count": 80000,
                "flow_duration": 4.0,
                "syn_flag_ratio": 0.99,
                "tcp_rst_ratio": 0.0,
            },
        ]

        batch_results = self.detector.predict_batch(flows)
        self.assertEqual(len(batch_results), 2)
        self.assertFalse(batch_results[0].is_anomaly)
        self.assertTrue(batch_results[1].is_anomaly)

    def test_model_save_and_load(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = os.path.join(tmpdir, "test_model.joblib")
            self.detector.save_model(model_path)
            self.assertTrue(os.path.exists(model_path))

            loaded = NetworkAnomalyDetector.load_model(model_path)
            self.assertTrue(loaded.is_fitted)

            test_flow = {
                "timestamp": 1700000000.0,
                "bytes_per_sec": 1000.0,
                "packets_per_sec": 10.0,
                "packet_count": 20,
                "byte_count": 2000,
                "flow_duration": 2.0,
                "syn_flag_ratio": 0.05,
                "tcp_rst_ratio": 0.0,
            }
            res_orig = self.detector.predict_flow(test_flow)
            res_loaded = loaded.predict_flow(test_flow)

            self.assertEqual(res_orig.is_anomaly, res_loaded.is_anomaly)
            self.assertAlmostEqual(res_orig.anomaly_score, res_loaded.anomaly_score, places=4)


if __name__ == "__main__":
    unittest.main()
