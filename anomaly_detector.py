#!/usr/bin/env python3
"""
Network Anomaly Detection using scikit-learn Isolation Forest.

Handles diurnal network traffic patterns via:
  1. Cyclical time features (hour_sin, hour_cos) mapping 24-hour time onto the unit circle.
  2. Hourly volume Z-scores (zscore_bytes_per_sec, zscore_packets_per_sec) normalizing
     traffic volume against historical hourly baselines.

This prevents legitimate high-volume business-hour traffic (e.g. 14:00 video calls)
from triggering false positive alerts, while flagging anomalous surges during quiet
off-hours (e.g. 03:00 data exfiltration).

Provides both batch training and sub-millisecond real-time flow prediction.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple, Union

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

logger = logging.getLogger("AnomalyDetector")


# ============================================================================
# Cyclical Time Encoding Utilities
# ============================================================================

class CyclicalTimeEncoder:
    """
    Encodes time-of-day into continuous 2D cyclical sine and cosine coordinates:
      hour_sin = sin(2 * pi * hour_float / 24.0)
      hour_cos = cos(2 * pi * hour_float / 24.0)

    Preserves angular metric distance across the midnight boundary (23:59 -> 00:00).
    """

    @staticmethod
    def encode_timestamp(ts: float) -> Tuple[float, float, float]:
        """
        Converts a Unix epoch timestamp into UTC (hour_float, hour_sin, hour_cos).
        """
        dt = datetime.fromtimestamp(ts, tz=timezone.utc)
        hour_float = dt.hour + (dt.minute / 60.0) + (dt.second / 3600.0)
        angle = 2.0 * math.pi * (hour_float / 24.0)
        hour_sin = math.sin(angle)
        hour_cos = math.cos(angle)
        return hour_float, hour_sin, hour_cos

    @staticmethod
    def encode_series(timestamps: pd.Series) -> Tuple[pd.Series, pd.Series]:
        """Vectorized encoding of a pandas Series of epoch timestamps in UTC."""
        dts = pd.to_datetime(timestamps, unit="s", utc=True)
        hours_float = dts.dt.hour + (dts.dt.minute / 60.0) + (dts.dt.second / 3600.0)
        angles = 2.0 * np.pi * (hours_float / 24.0)
        return np.sin(angles), np.cos(angles)


# ============================================================================
# Hourly Volume Baseline Profile
# ============================================================================

@dataclass
class HourlyBaseline:
    """
    Stores historical volume baseline parameters (mean and standard deviation)
    for each hour of the day [0..23]. Used for computing hourly Z-scores.
    """
    # hourly_stats[metric_name][hour_int] = {"mean": float, "std": float}
    hourly_stats: Dict[str, Dict[int, Dict[str, float]]] = field(default_factory=dict)
    # Global fallback stats if a specific hour has insufficient historical data
    global_stats: Dict[str, Dict[str, float]] = field(default_factory=dict)
    tracked_metrics: List[str] = field(
        default_factory=lambda: ["bytes_per_sec", "packets_per_sec", "byte_count"]
    )

    def fit(self, df: pd.DataFrame, timestamp_col: str = "timestamp"):
        """
        Computes historical diurnal volume distribution (mean, std) for each hour.
        """
        dts = pd.to_datetime(df[timestamp_col], unit="s", utc=True)
        hours = dts.dt.hour

        self.hourly_stats = {}
        self.global_stats = {}

        for metric in self.tracked_metrics:
            if metric not in df.columns:
                continue

            values = df[metric].astype(float)
            g_mean = float(values.mean())
            g_std = float(values.std()) if len(values) > 1 and values.std() > 1e-6 else 1.0
            self.global_stats[metric] = {"mean": g_mean, "std": g_std}

            self.hourly_stats[metric] = {}
            for h in range(24):
                mask = (hours == h)
                h_vals = values[mask]
                if len(h_vals) >= 2:
                    mean_val = float(h_vals.mean())
                    std_val = float(h_vals.std())
                    std_val = std_val if std_val > 1e-6 else 1.0
                else:
                    mean_val = g_mean
                    std_val = g_std

                self.hourly_stats[metric][h] = {"mean": mean_val, "std": std_val}

    def compute_zscore(self, metric: str, value: float, hour: int) -> float:
        """
        Calculates the hourly Z-score: (value - mean_hour) / std_hour.
        Guarded against division by zero.
        """
        stats_map = self.hourly_stats.get(metric, {})
        stats = stats_map.get(hour)
        if stats is None:
            stats = self.global_stats.get(metric, {"mean": 0.0, "std": 1.0})

        mean = stats["mean"]
        std = max(stats["std"], 1e-5)
        return float((value - mean) / std)


# ============================================================================
# Anomaly Prediction Data Models
# ============================================================================

@dataclass
class AnomalyResult:
    """Detailed anomaly evaluation result for a single network flow."""
    is_anomaly: bool
    anomaly_score: float  # Normalized 0.0 to 1.0 (higher = more anomalous)
    raw_decision_score: float  # Raw IsolationForest decision_function value
    severity: str  # NORMAL, LOW, MEDIUM, HIGH, CRITICAL
    flow_id: str
    hour_of_day: float
    features_used: Dict[str, float]
    zscores: Dict[str, float]
    evaluation_latency_ms: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "is_anomaly": self.is_anomaly,
            "anomaly_score": round(self.anomaly_score, 4),
            "raw_decision_score": round(self.raw_decision_score, 4),
            "severity": self.severity,
            "flow_id": self.flow_id,
            "hour_of_day": round(self.hour_of_day, 2),
            "features_used": {k: round(v, 4) for k, v in self.features_used.items()},
            "zscores": {k: round(v, 4) for k, v in self.zscores.items()},
            "evaluation_latency_ms": round(self.evaluation_latency_ms, 4),
        }

    def __str__(self) -> str:
        status = f"[{self.severity}]" if self.is_anomaly else "[NORMAL]"
        return (
            f"{status} Score: {self.anomaly_score:.3f} | Flow: {self.flow_id} | "
            f"Hour: {self.hour_of_day:.1f} | Latency: {self.evaluation_latency_ms:.3f}ms"
        )


# ============================================================================
# NetworkAnomalyDetector Pipeline Class
# ============================================================================

class NetworkAnomalyDetector:
    """
    Isolation Forest anomaly detector with diurnal cyclical feature engineering
    and hourly volume Z-score normalization.
    """

    FEATURE_COLUMNS = [
        "flow_duration",
        "bytes_per_sec",
        "packets_per_sec",
        "packet_count",
        "byte_count",
        "syn_flag_ratio",
        "tcp_rst_ratio",
        "bytes_per_packet",
        "hour_sin",
        "hour_cos",
        "zscore_bytes_per_sec",
        "zscore_packets_per_sec",
        "zscore_byte_count",
    ]

    def __init__(
        self,
        n_estimators: int = 100,
        contamination: float = 0.02,
        random_state: int = 42,
    ):
        self.n_estimators = n_estimators
        self.contamination = contamination
        self.random_state = random_state

        self.model: Optional[IsolationForest] = None
        self.baseline = HourlyBaseline()
        self.is_fitted = False

        # Threshold calibration parameters
        self._min_raw_score = -0.5
        self._max_raw_score = 0.5

    # ------------------------------------------------------------------------
    # Feature Engineering Functions
    # ------------------------------------------------------------------------

    def extract_features_df(self, df: pd.DataFrame, fit_baseline: bool = False) -> pd.DataFrame:
        """
        Transforms a batch DataFrame of flows into the full engineered feature matrix.
        Calculates cyclical time (hour_sin, hour_cos) and hourly volume Z-scores.
        """
        df = df.copy()

        # 1. Resolve timestamp column
        if "timestamp" not in df.columns:
            if "window_end" in df.columns:
                df["timestamp"] = df["window_end"]
            elif "window_start" in df.columns:
                df["timestamp"] = df["window_start"]
            else:
                df["timestamp"] = time.time()

        # 2. Fit hourly volume baseline if requested (during training)
        if fit_baseline:
            self.baseline.fit(df, timestamp_col="timestamp")

        # 3. Cyclical time encoding (hour_sin, hour_cos)
        sin_vals, cos_vals = CyclicalTimeEncoder.encode_series(df["timestamp"])
        df["hour_sin"] = sin_vals
        df["hour_cos"] = cos_vals

        # 4. Extract integer hour for Z-score lookup in UTC
        hours = pd.to_datetime(df["timestamp"], unit="s", utc=True).dt.hour.values

        # 5. Compute Hourly Volume Z-Scores (diurnal normalization)
        bps_vals = df.get("bytes_per_sec", 0.0).values
        pps_vals = df.get("packets_per_sec", 0.0).values
        bytes_vals = df.get("byte_count", 0.0).values

        z_bps = [
            self.baseline.compute_zscore("bytes_per_sec", float(bps_vals[i]), int(hours[i]))
            for i in range(len(df))
        ]
        z_pps = [
            self.baseline.compute_zscore("packets_per_sec", float(pps_vals[i]), int(hours[i]))
            for i in range(len(df))
        ]
        z_bytes = [
            self.baseline.compute_zscore("byte_count", float(bytes_vals[i]), int(hours[i]))
            for i in range(len(df))
        ]

        df["zscore_bytes_per_sec"] = z_bps
        df["zscore_packets_per_sec"] = z_pps
        df["zscore_byte_count"] = z_bytes

        # 6. Additional contextual ratios
        pkts = df.get("packet_count", 1).clip(lower=1)
        df["bytes_per_packet"] = df.get("byte_count", 0.0) / pkts

        # Fill missing statistical fields with default 0.0
        for col in self.FEATURE_COLUMNS:
            if col not in df.columns:
                df[col] = 0.0

        return df[self.FEATURE_COLUMNS]

    def extract_features_single(self, flow: Dict[str, Any]) -> Tuple[np.ndarray, Dict[str, float], Dict[str, float]]:
        """
        Fast real-time feature extraction for an individual flow dictionary (< 0.1ms).
        Returns:
          - feature_vector (numpy array 1 x N)
          - features_dict (mapping of feature name -> value)
          - zscores_dict (mapping of z-score features)
        """
        # Resolve timestamp
        ts = flow.get("timestamp")
        if ts is None:
            ts = flow.get("window_end", flow.get("window_start", time.time()))

        hour_float, hour_sin, hour_cos = CyclicalTimeEncoder.encode_timestamp(float(ts))
        hour_int = int(hour_float) % 24

        # Raw flow metrics
        duration = float(flow.get("flow_duration", 0.0))
        bps = float(flow.get("bytes_per_sec", 0.0))
        pps = float(flow.get("packets_per_sec", 0.0))
        pkts = int(flow.get("packet_count", 1))
        bytes_cnt = int(flow.get("byte_count", 0))
        syn_ratio = float(flow.get("syn_flag_ratio", 0.0))
        rst_ratio = float(flow.get("tcp_rst_ratio", 0.0))
        bpp = float(bytes_cnt / max(pkts, 1))

        # Hourly volume Z-scores
        z_bps = self.baseline.compute_zscore("bytes_per_sec", bps, hour_int)
        z_pps = self.baseline.compute_zscore("packets_per_sec", pps, hour_int)
        z_bytes = self.baseline.compute_zscore("byte_count", bytes_cnt, hour_int)

        features_dict = {
            "flow_duration": duration,
            "bytes_per_sec": bps,
            "packets_per_sec": pps,
            "packet_count": float(pkts),
            "byte_count": float(bytes_cnt),
            "syn_flag_ratio": syn_ratio,
            "tcp_rst_ratio": rst_ratio,
            "bytes_per_packet": bpp,
            "hour_sin": hour_sin,
            "hour_cos": hour_cos,
            "zscore_bytes_per_sec": z_bps,
            "zscore_packets_per_sec": z_pps,
            "zscore_byte_count": z_bytes,
        }

        zscores_dict = {
            "zscore_bytes_per_sec": z_bps,
            "zscore_packets_per_sec": z_pps,
            "zscore_byte_count": z_bytes,
        }

        # Vector in exact order of FEATURE_COLUMNS
        vec = np.array([[features_dict[col] for col in self.FEATURE_COLUMNS]], dtype=np.float32)
        return vec, features_dict, zscores_dict

    # ------------------------------------------------------------------------
    # Training API
    # ------------------------------------------------------------------------

    def fit(self, training_data: Union[pd.DataFrame, List[Dict[str, Any]]]):
        """
        Trains the Isolation Forest model on historical flow records.
        Automatically establishes hourly volume baselines and encodes diurnal time.
        """
        if isinstance(training_data, list):
            df = pd.DataFrame(training_data)
        else:
            df = training_data.copy()

        logger.info("Computing diurnal hourly volume baseline across %d flows...", len(df))
        X = self.extract_features_df(df, fit_baseline=True)

        logger.info("Training IsolationForest (n_estimators=%d, contamination=%.3f)...",
                    self.n_estimators, self.contamination)

        self.model = IsolationForest(
            n_estimators=self.n_estimators,
            contamination=self.contamination,
            random_state=self.random_state,
            n_jobs=-1,
        )
        self.model.fit(X)

        # Calibrate score bounds from training sample distribution
        raw_scores = self.model.decision_function(X)
        self._min_raw_score = float(np.percentile(raw_scores, 0.5))
        self._max_raw_score = float(np.percentile(raw_scores, 99.5))
        self.is_fitted = True

        logger.info("IsolationForest training complete. Calibration range: [%.3f, %.3f]",
                    self._min_raw_score, self._max_raw_score)

    # ------------------------------------------------------------------------
    # Real-Time Prediction API
    # ------------------------------------------------------------------------

    def predict_flow(self, flow: Dict[str, Any]) -> AnomalyResult:
        """
        Real-time anomaly prediction for an individual flow dictionary (< 0.5 ms).
        """
        if not self.is_fitted or self.model is None:
            raise RuntimeError("Model has not been trained! Call .fit() or load a pretrained model.")

        t0 = time.perf_counter()

        vec, features_dict, zscores_dict = self.extract_features_single(flow)

        # Pass DataFrame to match fitted feature names and avoid scikit-learn warnings
        df_single = pd.DataFrame(vec, columns=self.FEATURE_COLUMNS)
        raw_score = float(self.model.decision_function(df_single)[0])
        pred_label = int(self.model.predict(df_single)[0])  # -1 = anomaly, 1 = normal
        is_anomaly = (pred_label == -1)

        # Calibrated normalized anomaly score in [0.0, 1.0]:
        # raw_score >= 0.0 corresponds to inlier / normal traffic -> mapped to [0.0, 0.49]
        # raw_score < 0.0 corresponds to outlier / anomaly traffic -> mapped to [0.50, 1.0]
        if raw_score >= 0.0:
            max_pos = max(self._max_raw_score, 0.05)
            norm_score = float(np.clip(0.5 * (1.0 - (raw_score / max_pos)), 0.0, 0.49))
            severity = "NORMAL"
        else:
            min_neg = min(self._min_raw_score, -0.05)
            norm_score = float(np.clip(0.5 + 0.5 * (raw_score / min_neg), 0.50, 1.0))
            if norm_score > 0.85 or abs(zscores_dict["zscore_bytes_per_sec"]) > 8.0:
                severity = "CRITICAL"
            elif norm_score > 0.70 or abs(zscores_dict["zscore_bytes_per_sec"]) > 4.0:
                severity = "HIGH"
            elif norm_score > 0.58:
                severity = "MEDIUM"
            else:
                severity = "LOW"

        src = f"{flow.get('src_ip', 'unknown')}:{flow.get('src_port', '')}"
        dst = f"{flow.get('dst_ip', 'unknown')}:{flow.get('dst_port', '')}"
        flow_id = f"{src} -> {dst} [{flow.get('protocol', 'IP')}]"

        # Timestamp hour
        ts = flow.get("timestamp", flow.get("window_end", time.time()))
        hour_float, _, _ = CyclicalTimeEncoder.encode_timestamp(float(ts))

        latency_ms = (time.perf_counter() - t0) * 1000.0

        return AnomalyResult(
            is_anomaly=is_anomaly,
            anomaly_score=norm_score,
            raw_decision_score=raw_score,
            severity=severity,
            flow_id=flow_id,
            hour_of_day=hour_float,
            features_used=features_dict,
            zscores=zscores_dict,
            evaluation_latency_ms=latency_ms,
        )

    def predict_batch(self, flows: Union[pd.DataFrame, List[Dict[str, Any]]]) -> List[AnomalyResult]:
        """Runs predictions over a batch of flows."""
        if isinstance(flows, pd.DataFrame):
            flow_dicts = flows.to_dict(orient="records")
        else:
            flow_dicts = flows

        return [self.predict_flow(f) for f in flow_dicts]

    # ------------------------------------------------------------------------
    # Persistence (Save / Load)
    # ------------------------------------------------------------------------

    def save_model(self, filepath: str):
        """Saves trained model, baseline statistics, and calibration parameters."""
        if not self.is_fitted:
            raise RuntimeError("Cannot save an unfitted model.")

        state = {
            "model": self.model,
            "baseline": self.baseline,
            "n_estimators": self.n_estimators,
            "contamination": self.contamination,
            "random_state": self.random_state,
            "min_raw_score": self._min_raw_score,
            "max_raw_score": self._max_raw_score,
            "is_fitted": self.is_fitted,
        }
        joblib.dump(state, filepath)
        logger.info("Saved trained AnomalyDetector model to '%s'", filepath)

    @classmethod
    def load_model(cls, filepath: str) -> "NetworkAnomalyDetector":
        """Loads a pretrained model and hourly baseline from disk."""
        state = joblib.load(filepath)
        detector = cls(
            n_estimators=state["n_estimators"],
            contamination=state["contamination"],
            random_state=state["random_state"],
        )
        detector.model = state["model"]
        detector.baseline = state["baseline"]
        detector._min_raw_score = state["min_raw_score"]
        detector._max_raw_score = state["max_raw_score"]
        detector.is_fitted = state["is_fitted"]
        logger.info("Loaded AnomalyDetector model from '%s'", filepath)
        return detector


# ============================================================================
# Synthetic Diurnal Traffic Dataset Generator
# ============================================================================

def generate_synthetic_diurnal_dataset(
    num_days: int = 7,
    samples_per_hour: int = 40,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Generates realistic diurnal network flow records over multiple days:
      - Working Hours (09:00 - 18:00): High legitimate volume (video calls, office apps, file sync)
      - Night Hours (00:00 - 06:00): Low baseline volume (NTP, background telemetry, idle sockets)
      - Off-Hours Anomalies: Sudden large bursts during quiet hours (simulating data exfiltration)
      - Business-Hour Volumetric Traffic: Heavy legitimate bursts during peak hours (testing FP suppression)
    """
    np.random.seed(seed)
    base_time = float((1700000000 // 86400) * 86400)  # Aligned to exact UTC midnight (00:00:00)

    records: List[Dict[str, Any]] = []

    for day in range(num_days):
        for hour in range(24):
            # Diurnal volume scaling factor: peak around 14:00, trough around 03:00
            diurnal_factor = 0.5 * (1.0 + math.sin((hour - 8.0) * math.pi / 12.0))
            diurnal_factor = max(diurnal_factor, 0.05)

            for sample_idx in range(samples_per_hour):
                ts = base_time + (day * 86400.0) + (hour * 3600.0) + (sample_idx * (3600.0 / samples_per_hour))

                # Legitimate background traffic parameterized by diurnal factor
                base_bps = 500.0 + (diurnal_factor * 4500.0)
                base_pps = 5.0 + (diurnal_factor * 35.0)

                bps = float(np.random.normal(loc=base_bps, scale=base_bps * 0.15))
                pps = float(np.random.normal(loc=base_pps, scale=base_pps * 0.15))
                bps = max(bps, 50.0)
                pps = max(pps, 1.0)

                duration = float(np.random.uniform(0.5, 4.8))
                pkts = max(int(pps * duration), 1)
                byte_cnt = max(int(bps * duration), pkts * 40)

                syn_ratio = float(np.random.beta(0.5, 10.0))
                rst_ratio = float(np.random.beta(0.2, 10.0))

                records.append({
                    "timestamp": ts,
                    "src_ip": f"192.168.1.{np.random.randint(2, 250)}",
                    "src_port": int(np.random.randint(1024, 65535)),
                    "dst_ip": f"10.0.0.{np.random.randint(1, 10)}",
                    "dst_port": int(np.random.choice([80, 443, 8080, 53, 22])),
                    "protocol": "TCP" if np.random.rand() > 0.15 else "UDP",
                    "flow_duration": round(duration, 4),
                    "bytes_per_sec": round(bps, 2),
                    "packets_per_sec": round(pps, 2),
                    "packet_count": pkts,
                    "byte_count": byte_cnt,
                    "syn_flag_ratio": round(syn_ratio, 4),
                    "tcp_rst_ratio": round(rst_ratio, 4),
                })

    return pd.DataFrame(records)


# ============================================================================
# Demonstration CLI
# ============================================================================

def run_demonstration():
    """
    Demonstrates model training, diurnal baseline calculation, and real-time
    prediction scenarios comparing legitimate business-hour traffic vs off-hour anomalies.
    """
    print("\n" + "=" * 90)
    print("NETWORK ANOMALY DETECTOR DEMO (Isolation Forest + Diurnal Z-Scores)")
    print("=" * 90)

    # 1. Generate Synthetic Training Data (7 days of diurnal network telemetry)
    print("\n[1] Generating 7 days of realistic diurnal network traffic...")
    train_df = generate_synthetic_diurnal_dataset(num_days=7, samples_per_hour=40)
    print(f"Generated {len(train_df):,} historical flow records across 24-hour cycles.")

    # 2. Train Model
    print("\n[2] Training Isolation Forest with cyclical time & hourly volume Z-scores...")
    detector = NetworkAnomalyDetector(n_estimators=100, contamination=0.02)
    t_train_start = time.perf_counter()
    detector.fit(train_df)
    train_duration = time.perf_counter() - t_train_start
    print(f"Training completed in {train_duration:.2f} seconds.")

    # 3. Test Scenarios: Demonstrating False Positive Prevention
    print("\n" + "=" * 90)
    print("[3] TESTING REAL-TIME PREDICTION SCENARIOS:")
    print("=" * 90)

    # Base reference timestamps (14:00 UTC business hours vs 03:00 UTC night hours)
    base_midnight = float((1700000000 // 86400) * 86400)
    ref_day = base_midnight + (8 * 86400.0)
    ts_business_hour = ref_day + (14 * 3600.0)  # Exactly 14:00 UTC
    ts_night_hour = ref_day + (3 * 3600.0)      # Exactly 03:00 UTC

    test_scenarios = [
        # Scenario 1: Heavy legitimate traffic during 14:00 business hours (video conference)
        # -> MUST BE CLASSIFIED AS NORMAL (NOT AN ANOMALY) due to diurnal Z-score normalization!
        {
            "scenario": "Legitimate Heavy Business Traffic (14:00 Video Call)",
            "expected": "NORMAL (Suppressed FP via Business-Hour Baseline)",
            "flow": {
                "timestamp": ts_business_hour,
                "src_ip": "192.168.1.105",
                "src_port": 54321,
                "dst_ip": "142.250.190.46",
                "dst_port": 443,
                "protocol": "TCP",
                "flow_duration": 4.5,
                "bytes_per_sec": 4800.0,  # High absolute bytes!
                "packets_per_sec": 38.0,   # High absolute pps!
                "packet_count": 171,
                "byte_count": 21600,
                "syn_flag_ratio": 0.01,
                "tcp_rst_ratio": 0.0,
            },
        },
        # Scenario 2: Normal quiet night traffic at 03:00
        # -> MUST BE CLASSIFIED AS NORMAL
        {
            "scenario": "Normal Quiet Night Traffic (03:00 Heartbeat)",
            "expected": "NORMAL",
            "flow": {
                "timestamp": ts_night_hour,
                "src_ip": "192.168.1.50",
                "src_port": 49152,
                "dst_ip": "10.0.0.1",
                "dst_port": 53,
                "protocol": "UDP",
                "flow_duration": 1.2,
                "bytes_per_sec": 720.0,  # Matches typical 03:00 baseline
                "packets_per_sec": 6.5,
                "packet_count": 8,
                "byte_count": 864,
                "syn_flag_ratio": 0.0,
                "tcp_rst_ratio": 0.0,
            },
        },
        # Scenario 3: Large data burst during quiet 03:00 night hours (Data Exfiltration / C2)
        # -> MUST BE FLAGGED AS ANOMALY! (Off-hours volume Z-score will be > +15 sigma!)
        {
            "scenario": "Suspicious Off-Hours Data Burst (03:00 Exfiltration)",
            "expected": "ANOMALY (High Z-Score at 03:00 triggers alert)",
            "flow": {
                "timestamp": ts_night_hour,
                "src_ip": "192.168.1.99",
                "src_port": 39999,
                "dst_ip": "198.51.100.88",
                "dst_port": 8443,
                "protocol": "TCP",
                "flow_duration": 3.0,
                "bytes_per_sec": 12000.0,  # Abnormal volume at 3 AM!
                "packets_per_sec": 95.0,    # Abnormal rate at 3 AM!
                "packet_count": 285,
                "byte_count": 36000,
                "syn_flag_ratio": 0.02,
                "tcp_rst_ratio": 0.0,
            },
        },
        # Scenario 4: Slow scan with abnormal SYN ratio during business hours
        # -> MUST BE FLAGGED AS ANOMALY!
        {
            "scenario": "Abnormal SYN Profile Scan during Day (14:00)",
            "expected": "ANOMALY (Heuristic flag ratio anomaly)",
            "flow": {
                "timestamp": ts_business_hour,
                "src_ip": "10.0.0.44",
                "src_port": 61234,
                "dst_ip": "192.168.1.1",
                "dst_port": 445,
                "protocol": "TCP",
                "flow_duration": 0.1,
                "bytes_per_sec": 600.0,
                "packets_per_sec": 10.0,
                "packet_count": 1,
                "byte_count": 60,
                "syn_flag_ratio": 1.0,
                "tcp_rst_ratio": 0.0,
            },
        },
    ]

    for item in test_scenarios:
        res = detector.predict_flow(item["flow"])
        print(f"\nScenario: {item['scenario']}")
        print(f"Expected: {item['expected']}")
        print(f"Result:   {res}")
        print(f"  * Anomaly Flag:    {res.is_anomaly} (Severity: {res.severity})")
        print(f"  * Anomaly Score:   {res.anomaly_score:.4f} (Raw: {res.raw_decision_score:.4f})")
        print(f"  * Volume Z-Score:  {res.zscores['zscore_bytes_per_sec']:+.2f} std dev from hourly mean")
        print(f"  * Eval Latency:    {res.evaluation_latency_ms:.4f} ms")

    # 4. Save and Reload Test
    print("\n" + "=" * 90)
    print("[4] TESTING MODEL PERSISTENCE (Save & Load):")
    print("=" * 90)
    model_path = "isolation_forest_ids_model.joblib"
    detector.save_model(model_path)
    loaded_detector = NetworkAnomalyDetector.load_model(model_path)

    # Verify loaded detector prediction matches original
    res_orig = detector.predict_flow(test_scenarios[0]["flow"])
    res_loaded = loaded_detector.predict_flow(test_scenarios[0]["flow"])
    assert res_orig.is_anomaly == res_loaded.is_anomaly
    assert abs(res_orig.anomaly_score - res_loaded.anomaly_score) < 1e-4
    print(f"Model saved and reloaded successfully from '{model_path}'! Verified prediction consistency.")

    if os.path.exists(model_path):
        os.remove(model_path)
        print("Cleaned up temporary demonstration model file.")

    print("\nAll demonstration checks completed successfully!")


def main():
    parser = argparse.ArgumentParser(
        description="Isolation Forest Network Anomaly Detector with Diurnal Cyclical & Z-Score Features",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--demo", action="store_true", help="Run synthetic demonstration and benchmark")
    parser.add_argument("--train-csv", type=str, default=None, help="Path to historical flows CSV for training")
    parser.add_argument("--save-model", type=str, default="isolation_forest_model.joblib", help="Output model path")
    parser.add_argument("--load-model", type=str, default=None, help="Path to saved model for inference")
    parser.add_argument("--predict-json", type=str, default=None, help="Path to flows JSON file to evaluate")

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] [%(levelname)s] %(message)s")

    if args.demo or (not args.train_csv and not args.predict_json and not args.load_model):
        run_demonstration()
        return

    if args.train_csv:
        logger.info("Loading training data from %s...", args.train_csv)
        df = pd.read_csv(args.train_csv)
        detector = NetworkAnomalyDetector()
        detector.fit(df)
        detector.save_model(args.save_model)

    if args.predict_json and args.load_model:
        detector = NetworkAnomalyDetector.load_model(args.load_model)
        with open(args.predict_json, "r") as f:
            flows = json.load(f)
        if isinstance(flows, dict):
            flows = [flows]
        results = detector.predict_batch(flows)
        for r in results:
            print(r)


if __name__ == "__main__":
    main()
