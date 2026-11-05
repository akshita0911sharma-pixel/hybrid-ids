#!/usr/bin/env python3
"""
Unit and integration tests for Modular Scapy Network Flow Aggregator.
"""

import os
import queue
import tempfile
import threading
import time
import unittest

from scapy.layers.inet import ICMP, IP, TCP, UDP
from scapy.layers.inet6 import IPv6
from scapy.layers.l2 import ARP, Ether

from flow_aggregator import (
    FlowExporter,
    FlowKey,
    FlowMetrics,
    PacketParser,
    PacketRecord,
    SlidingWindowAggregator,
)


class TestFlowKey(unittest.TestCase):
    def test_flow_key_equality_and_hash(self):
        k1 = FlowKey("192.168.1.1", 1234, "10.0.0.1", 80, "TCP")
        k2 = FlowKey("192.168.1.1", 1234, "10.0.0.1", 80, "TCP")
        k3 = FlowKey("192.168.1.1", 1234, "10.0.0.1", 80, "UDP")

        self.assertEqual(k1, k2)
        self.assertNotEqual(k1, k3)
        self.assertEqual(hash(k1), hash(k2))

        # Check usable as dictionary key
        d = {k1: "flow1"}
        self.assertEqual(d[k2], "flow1")

    def test_to_bidirectional(self):
        forward = FlowKey("192.168.1.100", 50000, "93.184.216.34", 443, "TCP")
        reverse = FlowKey("93.184.216.34", 443, "192.168.1.100", 50000, "TCP")

        bi_forward = forward.to_bidirectional()
        bi_reverse = reverse.to_bidirectional()

        self.assertEqual(bi_forward, bi_reverse)
        self.assertEqual(bi_forward.src_ip, "192.168.1.100")
        self.assertEqual(bi_forward.dst_ip, "93.184.216.34")


class TestPacketParser(unittest.TestCase):
    def test_parse_tcp_syn(self):
        pkt = IP(src="192.168.1.5", dst="1.1.1.1") / TCP(sport=4321, dport=443, flags="S")
        record = PacketParser.parse(pkt)

        self.assertIsNotNone(record)
        self.assertEqual(record.flow_key.src_ip, "192.168.1.5")
        self.assertEqual(record.flow_key.dst_ip, "1.1.1.1")
        self.assertEqual(record.flow_key.src_port, 4321)
        self.assertEqual(record.flow_key.dst_port, 443)
        self.assertEqual(record.flow_key.protocol, "TCP")
        self.assertTrue(record.is_syn)
        self.assertFalse(record.is_rst)

    def test_parse_tcp_rst(self):
        pkt = IP(src="10.0.0.2", dst="10.0.0.1") / TCP(sport=8080, dport=55555, flags="R")
        record = PacketParser.parse(pkt)

        self.assertIsNotNone(record)
        self.assertFalse(record.is_syn)
        self.assertTrue(record.is_rst)

    def test_parse_tcp_syn_ack_and_rst_ack(self):
        pkt_sa = IP(src="10.0.0.2", dst="10.0.0.1") / TCP(sport=80, dport=55555, flags="SA")
        record_sa = PacketParser.parse(pkt_sa)
        self.assertTrue(record_sa.is_syn)
        self.assertFalse(record_sa.is_rst)

        pkt_ra = IP(src="10.0.0.2", dst="10.0.0.1") / TCP(sport=80, dport=55555, flags="RA")
        record_ra = PacketParser.parse(pkt_ra)
        self.assertFalse(record_ra.is_syn)
        self.assertTrue(record_ra.is_rst)

    def test_parse_udp(self):
        pkt = IP(src="192.168.1.20", dst="8.8.8.8") / UDP(sport=53000, dport=53)
        record = PacketParser.parse(pkt)

        self.assertIsNotNone(record)
        self.assertEqual(record.flow_key.protocol, "UDP")
        self.assertEqual(record.flow_key.src_port, 53000)
        self.assertEqual(record.flow_key.dst_port, 53)
        self.assertFalse(record.is_syn)
        self.assertFalse(record.is_rst)

    def test_parse_icmp(self):
        pkt = IP(src="192.168.1.20", dst="8.8.8.8") / ICMP()
        record = PacketParser.parse(pkt)

        self.assertIsNotNone(record)
        self.assertEqual(record.flow_key.protocol, "ICMP")
        self.assertEqual(record.flow_key.src_port, 0)
        self.assertEqual(record.flow_key.dst_port, 0)

    def test_parse_ipv6(self):
        pkt = IPv6(src="2001:db8::1", dst="2001:db8::2") / TCP(sport=5000, dport=80, flags="S")
        record = PacketParser.parse(pkt)

        self.assertIsNotNone(record)
        self.assertEqual(record.flow_key.src_ip, "2001:db8::1")
        self.assertEqual(record.flow_key.dst_ip, "2001:db8::2")
        self.assertEqual(record.flow_key.protocol, "TCP")
        self.assertTrue(record.is_syn)

    def test_parse_non_ip_packet(self):
        pkt = Ether() / ARP()
        record = PacketParser.parse(pkt)
        self.assertIsNone(record)


class TestSlidingWindowAggregator(unittest.TestCase):
    def setUp(self):
        self.queue = queue.Queue()
        self.shutdown_event = threading.Event()
        self.aggregator = SlidingWindowAggregator(
            packet_queue=self.queue,
            shutdown_event=self.shutdown_event,
            window_size=5.0,
            slide_interval=1.0,
            use_packet_time=True,
        )

    def test_metrics_calculation(self):
        key = FlowKey("1.2.3.4", 1000, "5.6.7.8", 80, "TCP")
        base = 100.0

        # Feed 4 packets:
        # t=100.0 (SYN), 100 bytes
        # t=101.0 (DATA), 300 bytes
        # t=101.5 (DATA), 200 bytes
        # t=102.0 (RST), 100 bytes
        # Total pkts: 4, Total bytes: 700, Duration: 2.0s
        # bytes_per_sec = 700 / 2.0 = 350.0
        # packets_per_sec = 4 / 2.0 = 2.0
        # syn_flag_ratio = 1 / 4 = 0.25
        # tcp_rst_ratio = 1 / 4 = 0.25
        packets = [
            PacketRecord(base + 0.0, 100, key, is_syn=True, is_rst=False),
            PacketRecord(base + 1.0, 300, key, is_syn=False, is_rst=False),
            PacketRecord(base + 1.5, 200, key, is_syn=False, is_rst=False),
            PacketRecord(base + 2.0, 100, key, is_syn=False, is_rst=True),
        ]
        for p in packets:
            self.aggregator._ingest_packet(p)

        metrics = self.aggregator.evaluate_window(current_time=base + 3.0)
        self.assertEqual(len(metrics), 1)

        m = metrics[0]
        self.assertEqual(m.flow_key, key)
        self.assertEqual(m.packet_count, 4)
        self.assertEqual(m.byte_count, 700)
        self.assertAlmostEqual(m.flow_duration, 2.0)
        self.assertAlmostEqual(m.bytes_per_sec, 350.0)
        self.assertAlmostEqual(m.packets_per_sec, 2.0)
        self.assertAlmostEqual(m.syn_flag_ratio, 0.25)
        self.assertAlmostEqual(m.tcp_rst_ratio, 0.25)

    def test_single_packet_flow_zero_division_guard(self):
        key = FlowKey("1.2.3.4", 1000, "5.6.7.8", 53, "UDP")
        self.aggregator._ingest_packet(PacketRecord(100.0, 80, key))

        metrics = self.aggregator.evaluate_window(current_time=101.0)
        self.assertEqual(len(metrics), 1)
        m = metrics[0]

        self.assertEqual(m.packet_count, 1)
        self.assertEqual(m.byte_count, 80)
        self.assertEqual(m.flow_duration, 0.0)
        self.assertEqual(m.bytes_per_sec, 0.0)
        self.assertEqual(m.packets_per_sec, 0.0)
        self.assertEqual(m.syn_flag_ratio, 0.0)
        self.assertEqual(m.tcp_rst_ratio, 0.0)

    def test_sliding_window_packet_eviction(self):
        key = FlowKey("1.1.1.1", 1111, "2.2.2.2", 2222, "TCP")

        # Window size is 5.0s
        # Packet at t=10.0 (will expire at t > 15.0)
        # Packet at t=13.0 (still valid at t=16.0)
        # Packet at t=15.5 (still valid at t=16.0)
        self.aggregator._ingest_packet(PacketRecord(10.0, 100, key))
        self.aggregator._ingest_packet(PacketRecord(13.0, 200, key))
        self.aggregator._ingest_packet(PacketRecord(15.5, 300, key))

        # Evaluate window at t=16.0: window is [11.0, 16.0]
        # Packet at 10.0 must be pruned!
        metrics = self.aggregator.evaluate_window(current_time=16.0)
        self.assertEqual(len(metrics), 1)
        m = metrics[0]

        self.assertEqual(m.packet_count, 2)
        self.assertEqual(m.byte_count, 500)
        self.assertAlmostEqual(m.flow_duration, 2.5)  # 15.5 - 13.0

        # Now evaluate window at t=21.0: window is [16.0, 21.0]
        # All packets are < 16.0, so flow must be completely evicted!
        metrics_expired = self.aggregator.evaluate_window(current_time=21.0)
        self.assertEqual(len(metrics_expired), 0)
        self.assertNotIn(key, self.aggregator.flow_buffers)


class TestFlowExporter(unittest.TestCase):
    def test_csv_and_json_export(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = os.path.join(tmpdir, "flows.csv")
            json_path = os.path.join(tmpdir, "flows.jsonl")

            exporter = FlowExporter(
                console_output=False,
                csv_path=csv_path,
                json_path=json_path,
            )

            metrics = [
                FlowMetrics(
                    flow_key=FlowKey("192.168.1.1", 5000, "10.0.0.1", 80, "TCP"),
                    window_start=100.0,
                    window_end=105.0,
                    packet_count=10,
                    byte_count=1500,
                    flow_duration=3.0,
                    bytes_per_sec=500.0,
                    packets_per_sec=3.33,
                    syn_flag_ratio=0.1,
                    tcp_rst_ratio=0.0,
                )
            ]

            exporter.export(metrics)
            exporter.close()

            # Verify CSV file contents
            self.assertTrue(os.path.exists(csv_path))
            with open(csv_path, "r", encoding="utf-8") as f:
                csv_content = f.read()
                self.assertIn("192.168.1.1", csv_content)
                self.assertIn("10.0.0.1", csv_content)
                self.assertIn("500.0", csv_content)

            # Verify JSON Lines file contents
            self.assertTrue(os.path.exists(json_path))
            with open(json_path, "r", encoding="utf-8") as f:
                json_content = f.read()
                self.assertIn('"src_ip": "192.168.1.1"', json_content)
                self.assertIn('"bytes_per_sec": 500.0', json_content)


class TestMultiThreadingLifecycle(unittest.TestCase):
    def test_queue_processing_and_graceful_shutdown(self):
        test_queue = queue.Queue()
        shutdown_event = threading.Event()
        received_batches = []

        def callback(metrics):
            received_batches.append(metrics)

        aggregator = SlidingWindowAggregator(
            packet_queue=test_queue,
            shutdown_event=shutdown_event,
            window_size=2.0,
            slide_interval=0.2,
            metrics_callback=callback,
        )

        aggregator.start()
        self.assertTrue(aggregator.is_alive())

        key = FlowKey("192.168.1.10", 1234, "192.168.1.20", 80, "TCP")
        now = time.time()
        test_queue.put(PacketRecord(now, 100, key, is_syn=True))
        test_queue.put(PacketRecord(now + 0.1, 200, key))

        # Allow consumer to process
        time.sleep(0.5)

        shutdown_event.set()
        aggregator.join(timeout=2.0)
        self.assertFalse(aggregator.is_alive())
        self.assertGreaterEqual(len(received_batches), 1)


if __name__ == "__main__":
    unittest.main()
