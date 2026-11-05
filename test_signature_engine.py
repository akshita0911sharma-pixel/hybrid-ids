#!/usr/bin/env python3
"""
Unit and benchmark test suite for SignatureEngine.
"""

import queue
import threading
import time
import unittest

from signature_engine import (
    Alert,
    AlertSeverity,
    SignatureEngine,
    SignatureEngineWorker,
    SignatureRule,
    StatefulPortScanDetector,
)


class TestSignatureEngine(unittest.TestCase):
    def setUp(self):
        self.secondary_q = queue.Queue()
        self.alerts_q = queue.Queue()
        self.captured_alerts = []

        def callback(alert: Alert):
            self.captured_alerts.append(alert)

        self.engine = SignatureEngine(
            secondary_queue=self.secondary_q,
            alerts_queue=self.alerts_q,
            alert_callback=callback,
            enable_stateful_scan_detection=True,
            scan_port_threshold=5,  # Lower threshold for faster test assertion
        )

    # ------------------------------------------------------------------------
    # 1. AlertSeverity Tests
    # ------------------------------------------------------------------------

    def test_severity_ordering(self):
        self.assertLess(AlertSeverity.LOW, AlertSeverity.MEDIUM)
        self.assertLess(AlertSeverity.MEDIUM, AlertSeverity.HIGH)
        self.assertLess(AlertSeverity.HIGH, AlertSeverity.CRITICAL)

    # ------------------------------------------------------------------------
    # 2. SYN Flood Detection
    # ------------------------------------------------------------------------

    def test_syn_flood_detected(self):
        flood_flow = {
            "src_ip": "10.0.0.15",
            "src_port": 49152,
            "dst_ip": "192.168.1.1",
            "dst_port": 80,
            "protocol": "TCP",
            "packet_count": 50,
            "byte_count": 3000,
            "flow_duration": 1.2,
            "bytes_per_sec": 2500.0,
            "packets_per_sec": 41.6,
            "syn_flag_ratio": 0.96,
            "tcp_rst_ratio": 0.0,
        }

        alert, matched = self.engine.evaluate(flood_flow)

        self.assertTrue(matched)
        self.assertIsNotNone(alert)
        self.assertEqual(alert.rule_id, "RULE-DOS-SYN-FLOOD")
        self.assertEqual(alert.severity, AlertSeverity.CRITICAL)
        self.assertEqual(len(self.captured_alerts), 1)
        self.assertEqual(self.secondary_q.qsize(), 0)

    def test_normal_tcp_not_flagged_as_syn_flood(self):
        normal_flow = {
            "src_ip": "192.168.1.50",
            "src_port": 52100,
            "dst_ip": "192.168.1.1",
            "dst_port": 80,
            "protocol": "TCP",
            "packet_count": 20,
            "byte_count": 14000,
            "flow_duration": 2.5,
            "bytes_per_sec": 5600.0,
            "packets_per_sec": 8.0,
            "syn_flag_ratio": 0.05,
            "tcp_rst_ratio": 0.0,
        }

        alert, matched = self.engine.evaluate(normal_flow)

        self.assertFalse(matched)
        self.assertIsNone(alert)
        self.assertEqual(len(self.captured_alerts), 0)
        self.assertEqual(self.secondary_q.qsize(), 1)
        passed_flow = self.secondary_q.get()
        self.assertEqual(passed_flow["src_ip"], "192.168.1.50")

    # ------------------------------------------------------------------------
    # 3. Port Scan Detection Tests
    # ------------------------------------------------------------------------

    def test_stateful_rapid_port_scan_detection(self):
        scanner_ip = "192.0.2.1"
        target_ip = "10.0.0.5"

        # Probe 5 different ports (threshold is 5)
        for i, port in enumerate([21, 22, 23, 25, 80]):
            flow = {
                "src_ip": scanner_ip,
                "src_port": 40000 + i,
                "dst_ip": target_ip,
                "dst_port": port,
                "protocol": "TCP",
                "packet_count": 1,
                "flow_duration": 0.0,
                "syn_flag_ratio": 1.0,
                "tcp_rst_ratio": 0.0,
            }
            alert, matched = self.engine.evaluate(flow)
            if i < 4:
                # May match stealth single probe or pass through
                pass
            else:
                # 5th distinct port must trigger stateful rapid scan
                self.assertTrue(matched)
                self.assertIsNotNone(alert)
                self.assertEqual(alert.rule_id, "RULE-RECON-RAPID-PORTSCAN")
                self.assertEqual(alert.severity, AlertSeverity.HIGH)

    # ------------------------------------------------------------------------
    # 4. Suspicious Payload Signatures
    # ------------------------------------------------------------------------

    def test_sqli_payload_detection(self):
        sqli_flow = {
            "src_ip": "10.0.0.8",
            "src_port": 34567,
            "dst_ip": "192.168.1.100",
            "dst_port": 80,
            "protocol": "TCP",
            "packet_count": 4,
            "flow_duration": 0.5,
            "syn_flag_ratio": 0.25,
            "tcp_rst_ratio": 0.0,
            "payload": b"GET /users?id=1' UNION SELECT username,password FROM users-- HTTP/1.1\r\n",
        }

        alert, matched = self.engine.evaluate(sqli_flow)

        self.assertTrue(matched)
        self.assertIsNotNone(alert)
        self.assertEqual(alert.rule_id, "RULE-APP-SQLI-INJECTION")
        self.assertEqual(alert.severity, AlertSeverity.HIGH)
        self.assertIn("union select", alert.matched_detail.lower())

    def test_rce_shell_payload_detection(self):
        rce_flow = {
            "src_ip": "10.0.0.9",
            "src_port": 34568,
            "dst_ip": "192.168.1.100",
            "dst_port": 8080,
            "protocol": "TCP",
            "packet_count": 3,
            "flow_duration": 0.2,
            "syn_flag_ratio": 0.0,
            "tcp_rst_ratio": 0.0,
            "payload": "POST /api/exec HTTP/1.1\r\n\r\ncat /etc/passwd",
        }

        alert, matched = self.engine.evaluate(rce_flow)

        self.assertTrue(matched)
        self.assertIsNotNone(alert)
        self.assertEqual(alert.rule_id, "RULE-APP-RCE-SHELL")
        self.assertEqual(alert.severity, AlertSeverity.CRITICAL)

    # ------------------------------------------------------------------------
    # 5. Custom Rule Management Tests
    # ------------------------------------------------------------------------

    def test_add_and_disable_custom_rule(self):
        custom_rule = SignatureRule(
            rule_id="RULE-CUSTOM-DNS-TUNNEL",
            name="Suspected DNS Tunneling Rate",
            severity=AlertSeverity.HIGH,
            category="Exfiltration",
            description="Abnormally large UDP flow volume on DNS port 53",
            protocol="UDP",
            target_ports={53},
            min_packet_count=100,
            min_bytes_per_sec=10000.0,
        )

        self.engine.add_rule(custom_rule)
        retrieved = self.engine.get_rule("RULE-CUSTOM-DNS-TUNNEL")
        self.assertIsNotNone(retrieved)

        dns_flow = {
            "src_ip": "192.168.1.75",
            "src_port": 55555,
            "dst_ip": "8.8.8.8",
            "dst_port": 53,
            "protocol": "UDP",
            "packet_count": 150,
            "flow_duration": 2.0,
            "bytes_per_sec": 15000.0,
        }

        alert, matched = self.engine.evaluate(dns_flow)
        self.assertTrue(matched)
        self.assertEqual(alert.rule_id, "RULE-CUSTOM-DNS-TUNNEL")

        # Disable the rule
        self.engine.enable_rule("RULE-CUSTOM-DNS-TUNNEL", enabled=False)
        alert2, matched2 = self.engine.evaluate(dns_flow)
        self.assertFalse(matched2)

    # ------------------------------------------------------------------------
    # 6. Secondary Queue Routing
    # ------------------------------------------------------------------------

    def test_routing_clean_traffic_to_secondary_queue(self):
        clean_udp = {
            "src_ip": "192.168.1.10",
            "src_port": 12345,
            "dst_ip": "1.1.1.1",
            "dst_port": 53,
            "protocol": "UDP",
            "packet_count": 2,
            "flow_duration": 0.05,
            "bytes_per_sec": 1200.0,
        }

        alert, matched = self.engine.evaluate(clean_udp)
        self.assertFalse(matched)
        self.assertIsNone(alert)
        self.assertEqual(self.secondary_q.qsize(), 1)
        self.assertEqual(self.alerts_q.qsize(), 0)

    # ------------------------------------------------------------------------
    # 7. Latency Requirement Verification (< 2ms)
    # ------------------------------------------------------------------------

    def test_latency_under_2ms_per_flow(self):
        sample_flow = {
            "src_ip": "192.168.1.200",
            "src_port": 45000,
            "dst_ip": "10.0.0.1",
            "dst_port": 443,
            "protocol": "TCP",
            "packet_count": 10,
            "byte_count": 4000,
            "flow_duration": 1.5,
            "bytes_per_sec": 2666.6,
            "packets_per_sec": 6.66,
            "syn_flag_ratio": 0.1,
            "tcp_rst_ratio": 0.0,
            "payload": b"GET /api/v1/status HTTP/1.1\r\nHost: internal\r\n\r\n",
        }

        # Warm-up
        for _ in range(50):
            self.engine.evaluate(sample_flow)

        # Measure 1,000 flows
        latencies = []
        for _ in range(1000):
            t0 = time.perf_counter()
            self.engine.evaluate(sample_flow)
            latencies.append((time.perf_counter() - t0) * 1000.0)

        avg_latency = sum(latencies) / len(latencies)
        sorted_lat = sorted(latencies)
        p99_latency = sorted_lat[int(len(sorted_lat) * 0.99)]

        print(f"\n[TestLatency] Avg: {avg_latency:.4f} ms | p99: {p99_latency:.4f} ms")
        self.assertLess(avg_latency, 0.5, "Average latency must be well under 0.5ms")
        self.assertLess(p99_latency, 2.0, "p99 latency must be under 2.0ms requirement")

    # ------------------------------------------------------------------------
    # 8. Threaded Worker Lifecycle
    # ------------------------------------------------------------------------

    def test_threaded_worker_continuous_processing(self):
        in_q = queue.Queue()
        out_q = queue.Queue()
        alerts_q = queue.Queue()
        shutdown = threading.Event()

        engine = SignatureEngine(secondary_queue=out_q, alerts_queue=alerts_q)
        worker = SignatureEngineWorker(engine, in_q, shutdown)
        worker.start()

        # Enqueue 10 clean flows and 1 SYN flood flow
        for i in range(10):
            in_q.put({
                "src_ip": f"10.0.0.{i}",
                "src_port": 5000,
                "dst_ip": "8.8.8.8",
                "dst_port": 53,
                "protocol": "UDP",
                "packet_count": 1,
                "flow_duration": 0.0,
            })

        in_q.put({
            "src_ip": "1.2.3.4",
            "src_port": 11111,
            "dst_ip": "10.0.0.1",
            "dst_port": 80,
            "protocol": "TCP",
            "packet_count": 50,
            "flow_duration": 1.0,
            "packets_per_sec": 50.0,
            "syn_flag_ratio": 1.0,
        })

        time.sleep(0.3)
        shutdown.set()
        worker.join(timeout=2.0)

        self.assertFalse(worker.is_alive())
        self.assertEqual(out_q.qsize(), 10, "10 clean flows routed to secondary queue")
        self.assertEqual(alerts_q.qsize(), 1, "1 flood alert captured")


if __name__ == "__main__":
    unittest.main()
