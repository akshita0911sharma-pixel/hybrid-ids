#!/usr/bin/env python3
"""
Unit and integration test suite for Central Hybrid IDS Orchestrator.
"""

import io
import json
import os
import queue
import tempfile
import threading
import time
import unittest

from anomaly_detector import NetworkAnomalyDetector, generate_synthetic_diurnal_dataset
from orchestrator import (
    AlertAggregator,
    AlertAggregatorWorker,
    HybridIDSOrchestrator,
    Stage1SignatureWorker,
    Stage2AnomalyWorker,
    UnifiedAlert,
)
from signature_engine import SignatureEngine


class TestAlertAggregator(unittest.TestCase):
    def setUp(self):
        self.output_io = io.StringIO()
        self.aggregator = AlertAggregator(
            dedup_window_seconds=10.0,
            console_stream=False,
            output_io=self.output_io,
        )

    def _get_emitted_records(self):
        self.output_io.seek(0)
        lines = [line.strip() for line in self.output_io.getvalue().split("\n") if line.strip()]
        return [json.loads(line) for line in lines]

    def test_single_alert_emitted_immediately(self):
        alert = UnifiedAlert(
            alert_id="ALT-1",
            stage="STAGE_1_SIGNATURE",
            severity="HIGH",
            category="Reconnaissance",
            rule_or_model="PortScan",
            description="SYN scan detected",
            source_ip="192.168.1.50",
            source_port=12345,
            dest_ip="10.0.0.1",
            dest_port=80,
            protocol="TCP",
            flow_metrics={"packet_count": 1},
            detection_details={"detail": "syn=1"},
            timestamp=100.0,
        )

        emitted = self.aggregator.process_alert(alert)
        self.assertTrue(emitted)

        records = self._get_emitted_records()
        self.assertEqual(len(records), 1)
        r = records[0]
        self.assertEqual(r["event_type"], "ALERT")
        self.assertEqual(r["source_ip"], "192.168.1.50")
        self.assertEqual(r["severity"], "HIGH")
        self.assertTrue(r["dedup_metadata"]["is_first_in_window"])

    def test_duplicate_alerts_suppressed_within_window(self):
        # 1st alert at t=100.0
        alert1 = UnifiedAlert(
            alert_id="ALT-1",
            stage="STAGE_1_SIGNATURE",
            severity="HIGH",
            category="Recon",
            rule_or_model="PortScan",
            description="Test scan",
            source_ip="10.0.0.5",
            source_port=1000,
            dest_ip="1.1.1.1",
            dest_port=80,
            protocol="TCP",
            flow_metrics={},
            detection_details={},
            timestamp=100.0,
        )
        self.assertTrue(self.aggregator.process_alert(alert1))

        # 2nd alert from same IP at t=102.0 -> suppressed
        alert2 = UnifiedAlert(
            alert_id="ALT-2",
            stage="STAGE_1_SIGNATURE",
            severity="HIGH",
            category="Recon",
            rule_or_model="PortScan",
            description="Test scan",
            source_ip="10.0.0.5",
            source_port=1001,
            dest_ip="1.1.1.1",
            dest_port=81,
            protocol="TCP",
            flow_metrics={},
            detection_details={},
            timestamp=102.0,
        )
        self.assertFalse(self.aggregator.process_alert(alert2))

        # 3rd alert from same IP at t=104.0 -> suppressed
        alert3 = UnifiedAlert(
            alert_id="ALT-3",
            stage="STAGE_1_SIGNATURE",
            severity="HIGH",
            category="Recon",
            rule_or_model="PortScan",
            description="Test scan",
            source_ip="10.0.0.5",
            source_port=1002,
            dest_ip="1.1.1.1",
            dest_port=82,
            protocol="TCP",
            flow_metrics={},
            detection_details={},
            timestamp=104.0,
        )
        self.assertFalse(self.aggregator.process_alert(alert3))

        records = self._get_emitted_records()
        self.assertEqual(len(records), 1, "Only initial alert emitted immediately")
        self.assertEqual(self.aggregator.total_alerts_received, 3)
        self.assertEqual(self.aggregator.total_alerts_suppressed, 2)

    def test_window_expiration_triggers_aggregated_summary(self):
        # 3 alerts within window [100.0 .. 110.0]
        for i in range(3):
            alert = UnifiedAlert(
                alert_id=f"ALT-{i}",
                stage="STAGE_1_SIGNATURE",
                severity="MEDIUM",
                category="Recon",
                rule_or_model="Probe",
                description="Probe",
                source_ip="172.16.0.9",
                source_port=5000 + i,
                dest_ip="10.0.0.1",
                dest_port=80,
                protocol="TCP",
                flow_metrics={},
                detection_details={},
                timestamp=100.0 + i,
            )
            self.aggregator.process_alert(alert)

        # Advance time to t=111.0 (window expired: 111.0 - 100.0 = 11.0 >= 10.0)
        flushed = self.aggregator.check_expired_windows(current_time=111.0)
        self.assertEqual(flushed, 1)

        records = self._get_emitted_records()
        self.assertEqual(len(records), 2)
        summary = records[1]
        self.assertEqual(summary["event_type"], "AGGREGATED_SUMMARY")
        self.assertEqual(summary["source_ip"], "172.16.0.9")
        self.assertEqual(summary["total_alerts_in_window"], 3)
        self.assertEqual(summary["total_alerts_suppressed"], 2)

    def test_severity_escalation_triggers_immediate_notice(self):
        # Initial alert at t=100.0: LOW severity
        alert_low = UnifiedAlert(
            alert_id="ALT-LOW",
            stage="STAGE_1_SIGNATURE",
            severity="LOW",
            category="Recon",
            rule_or_model="RST-Sweep",
            description="Connection Reset",
            source_ip="192.168.1.80",
            source_port=2000,
            dest_ip="10.0.0.1",
            dest_port=80,
            protocol="TCP",
            flow_metrics={},
            detection_details={},
            timestamp=100.0,
        )
        self.aggregator.process_alert(alert_low)

        # Escalating alert at t=102.0: CRITICAL severity
        alert_crit = UnifiedAlert(
            alert_id="ALT-CRIT",
            stage="STAGE_1_SIGNATURE",
            severity="CRITICAL",
            category="RCE",
            rule_or_model="ShellExecution",
            description="Remote Shell Command",
            source_ip="192.168.1.80",
            source_port=2001,
            dest_ip="10.0.0.1",
            dest_port=8080,
            protocol="TCP",
            flow_metrics={},
            detection_details={},
            timestamp=102.0,
        )
        self.aggregator.process_alert(alert_crit)

        records = self._get_emitted_records()
        self.assertEqual(len(records), 2)
        self.assertEqual(records[0]["event_type"], "ALERT")
        self.assertEqual(records[0]["severity"], "LOW")
        self.assertEqual(records[1]["event_type"], "SEVERITY_ESCALATION")
        self.assertEqual(records[1]["escalated_severity"], "CRITICAL")


class TestOrchestratorPipeline(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Prepare trained anomaly detector
        train_df = generate_synthetic_diurnal_dataset(num_days=7, samples_per_hour=25, seed=42)
        cls.anomaly_detector = NetworkAnomalyDetector(n_estimators=100, contamination=0.02, random_state=42)
        cls.anomaly_detector.fit(train_df)

    def setUp(self):
        self.output_io = io.StringIO()
        self.orchestrator = HybridIDSOrchestrator(
            dedup_window_seconds=5.0,
            console_json_stream=False,
            alert_log_file=None,
        )
        # Inject output_io into orchestrator's aggregator
        self.orchestrator.initialize_engines()
        self.orchestrator.alert_aggregator.output_io = self.output_io
        self.orchestrator.anomaly_detector = self.anomaly_detector

        # Start workers
        self.orchestrator.stage1_worker = Stage1SignatureWorker(
            signature_engine=self.orchestrator.signature_engine,
            raw_flow_queue=self.orchestrator.raw_flow_queue,
            alert_ingress_queue=self.orchestrator.alert_ingress_queue,
            shutdown_event=self.orchestrator.shutdown_event,
        )
        self.orchestrator.stage2_worker = Stage2AnomalyWorker(
            anomaly_detector=self.orchestrator.anomaly_detector,
            stage2_queue=self.orchestrator.stage2_queue,
            alert_ingress_queue=self.orchestrator.alert_ingress_queue,
            shutdown_event=self.orchestrator.shutdown_event,
        )
        self.orchestrator.aggregator_worker = AlertAggregatorWorker(
            aggregator=self.orchestrator.alert_aggregator,
            alert_ingress_queue=self.orchestrator.alert_ingress_queue,
            shutdown_event=self.orchestrator.shutdown_event,
        )

        self.orchestrator.aggregator_worker.start()
        self.orchestrator.stage2_worker.start()
        self.orchestrator.stage1_worker.start()

    def tearDown(self):
        self.orchestrator.stop()

    def _get_records(self):
        self.output_io.seek(0)
        lines = [line.strip() for line in self.output_io.getvalue().split("\n") if line.strip()]
        return [json.loads(line) for line in lines]

    def test_normal_traffic_no_alerts(self):
        base_midnight = float((1700000000 // 86400) * 86400)
        ts_1400 = base_midnight + (8 * 86400.0) + (14 * 3600.0)

        # Enqueue legitimate daytime traffic
        self.orchestrator.enqueue_flow({
            "timestamp": ts_1400,
            "src_ip": "192.168.1.15",
            "src_port": 50000,
            "dst_ip": "93.184.216.34",
            "dst_port": 443,
            "protocol": "TCP",
            "flow_duration": 4.0,
            "bytes_per_sec": 4900.0,
            "packets_per_sec": 39.0,
            "packet_count": 156,
            "byte_count": 19600,
            "syn_flag_ratio": 0.01,
            "tcp_rst_ratio": 0.0,
        })

        time.sleep(0.4)
        records = self._get_records()
        self.assertEqual(len(records), 0, "Legitimate business traffic must yield zero alerts")

    def test_stage1_match_routes_alert_and_skips_stage2(self):
        base_midnight = float((1700000000 // 86400) * 86400)
        ts_1400 = base_midnight + (8 * 86400.0) + (14 * 3600.0)

        # Enqueue SYN flood
        self.orchestrator.enqueue_flow({
            "timestamp": ts_1400,
            "src_ip": "10.0.0.99",
            "src_port": 45000,
            "dst_ip": "192.168.1.1",
            "dst_port": 80,
            "protocol": "TCP",
            "flow_duration": 1.0,
            "bytes_per_sec": 2000.0,
            "packets_per_sec": 30.0,
            "packet_count": 30,
            "byte_count": 2000,
            "syn_flag_ratio": 0.95,
            "tcp_rst_ratio": 0.0,
        })

        time.sleep(0.4)
        records = self._get_records()
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]["stage"], "STAGE_1_SIGNATURE")
        self.assertEqual(records[0]["severity"], "CRITICAL")
        self.assertEqual(records[0]["source_ip"], "10.0.0.99")
        # Ensure Stage 2 never received this flow
        self.assertEqual(self.orchestrator.stage2_worker.processed_count, 0)

    def test_clean_flow_reaches_stage2_and_flags_anomaly(self):
        base_midnight = float((1700000000 // 86400) * 86400)
        ts_0300 = base_midnight + (8 * 86400.0) + (3 * 3600.0)

        # Enqueue 03:00 off-hours data burst (no signature pattern, but huge volumetric burst)
        self.orchestrator.enqueue_flow({
            "timestamp": ts_0300,
            "src_ip": "192.168.1.200",
            "src_port": 39999,
            "dst_ip": "198.51.100.88",
            "dst_port": 8443,
            "protocol": "TCP",
            "flow_duration": 3.0,
            "bytes_per_sec": 15000.0,
            "packets_per_sec": 100.0,
            "packet_count": 300,
            "byte_count": 45000,
            "syn_flag_ratio": 0.0,
            "tcp_rst_ratio": 0.0,
        })

        time.sleep(0.4)
        records = self._get_records()
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]["stage"], "STAGE_2_ANOMALY")
        self.assertEqual(records[0]["severity"], "CRITICAL")
        self.assertEqual(records[0]["source_ip"], "192.168.1.200")
        self.assertIn("anomaly_score", records[0]["detection_details"])


if __name__ == "__main__":
    unittest.main()
