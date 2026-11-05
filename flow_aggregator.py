#!/usr/bin/env python3
"""
Modular Network Flow Aggregator using Scapy.

Captures live network packets from a specified interface (or simulated traffic),
groups them into 5-tuple network flows (Source IP, Source Port, Dest IP, Dest Port, Protocol),
and computes flow metrics across a 5-second sliding window:
  - flow_duration (seconds)
  - bytes_per_sec (throughput)
  - packets_per_sec (rate)
  - syn_flag_ratio (ratio of TCP SYN packets)
  - tcp_rst_ratio (ratio of TCP RST packets)

Architecture:
  - PacketCaptureWorker: Background thread running Scapy sniff with robust error
    handling and non-blocking queue ingestion.
  - SlidingWindowAggregator: Background thread managing active flow deques,
    pruning expired packets outside the 5s window, and extracting features.
  - FlowMonitor: Manager class handling thread synchronization, signal handling,
    and callback notifications.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import queue
import signal
import sys
import threading
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime
from typing import Callable, Deque, Dict, Iterator, List, Optional, Tuple

# Suppress Scapy IPv6 warning banner on startup
logging.getLogger("scapy.runtime").setLevel(logging.ERROR)

try:
    import scapy.all as scapy
    from scapy.layers.inet import ICMP, IP, TCP, UDP
    from scapy.layers.inet6 import IPv6
except ImportError:
    scapy = None
    IP = None
    IPv6 = None
    TCP = None
    UDP = None
    ICMP = None


# Configure application logger
logger = logging.getLogger("FlowAggregator")


# ============================================================================
# Data Models
# ============================================================================

@dataclass(frozen=True)
class FlowKey:
    """
    Represents a standard 5-tuple identifying a unidirectional or bidirectional flow.
    """
    src_ip: str
    src_port: int
    dst_ip: str
    dst_port: int
    protocol: str

    def __str__(self) -> str:
        return f"{self.src_ip}:{self.src_port} -> {self.dst_ip}:{self.dst_port} [{self.protocol}]"

    def to_bidirectional(self) -> "FlowKey":
        """
        Normalizes flow endpoints so A -> B and B -> A map to the exact same flow key.
        """
        endpoint_a = (self.src_ip, self.src_port)
        endpoint_b = (self.dst_ip, self.dst_port)
        if endpoint_a <= endpoint_b:
            return self
        return FlowKey(
            src_ip=self.dst_ip,
            src_port=self.dst_port,
            dst_ip=self.src_ip,
            dst_port=self.src_port,
            protocol=self.protocol,
        )


@dataclass(frozen=True)
class PacketRecord:
    """
    Lightweight internal representation of a captured packet.
    Decoupled from heavy Scapy Packet objects to conserve memory and minimize queue latency.
    """
    timestamp: float
    length: int
    flow_key: FlowKey
    is_syn: bool = False
    is_rst: bool = False


@dataclass
class FlowMetrics:
    """
    Aggregated statistical metrics for a single 5-tuple flow within a sliding window.
    """
    flow_key: FlowKey
    window_start: float
    window_end: float
    packet_count: int
    byte_count: int
    flow_duration: float
    bytes_per_sec: float
    packets_per_sec: float
    syn_flag_ratio: float
    tcp_rst_ratio: float

    def to_dict(self) -> dict:
        """Serializes the metrics into a standard dictionary."""
        return {
            "src_ip": self.flow_key.src_ip,
            "src_port": self.flow_key.src_port,
            "dst_ip": self.flow_key.dst_ip,
            "dst_port": self.flow_key.dst_port,
            "protocol": self.flow_key.protocol,
            "window_start": round(self.window_start, 3),
            "window_end": round(self.window_end, 3),
            "packet_count": self.packet_count,
            "byte_count": self.byte_count,
            "flow_duration": round(self.flow_duration, 4),
            "bytes_per_sec": round(self.bytes_per_sec, 2),
            "packets_per_sec": round(self.packets_per_sec, 2),
            "syn_flag_ratio": round(self.syn_flag_ratio, 4),
            "tcp_rst_ratio": round(self.tcp_rst_ratio, 4),
        }


# ============================================================================
# Packet Parsing Utilities
# ============================================================================

class PacketParser:
    """
    Extracts 5-tuple information and protocol flags from raw Scapy packets.
    """

    # Protocol number to human-readable string mapping
    PROTO_MAP = {
        1: "ICMP",
        6: "TCP",
        17: "UDP",
        58: "ICMPv6",
    }

    @staticmethod
    def parse(packet, bidirectional: bool = False) -> Optional[PacketRecord]:
        """
        Parses a Scapy packet into a lightweight PacketRecord.
        Returns None if packet is not an IP packet or lacks transport headers.
        """
        try:
            # 1. Extract Network Layer (IPv4 or IPv6)
            if packet.haslayer(IP):
                ip_layer = packet[IP]
                src_ip = str(ip_layer.src)
                dst_ip = str(ip_layer.dst)
                proto_num = int(ip_layer.proto)
            elif packet.haslayer(IPv6):
                ip_layer = packet[IPv6]
                src_ip = str(ip_layer.src)
                dst_ip = str(ip_layer.dst)
                proto_num = int(ip_layer.nh)
            else:
                return None  # Non-IP packet (ARP, STP, LLC, etc.)

            # 2. Extract Transport Layer
            src_port = 0
            dst_port = 0
            is_syn = False
            is_rst = False
            proto_name = PacketParser.PROTO_MAP.get(proto_num, f"IP-{proto_num}")

            if packet.haslayer(TCP):
                tcp_layer = packet[TCP]
                src_port = int(tcp_layer.sport)
                dst_port = int(tcp_layer.dport)
                proto_name = "TCP"

                # Check TCP Flags via bitmask: SYN is 0x02, RST is 0x04
                try:
                    flags_int = int(tcp_layer.flags)
                    is_syn = bool(flags_int & 0x02)
                    is_rst = bool(flags_int & 0x04)
                except (ValueError, TypeError):
                    flags_str = str(tcp_layer.flags)
                    is_syn = "S" in flags_str
                    is_rst = "R" in flags_str

            elif packet.haslayer(UDP):
                udp_layer = packet[UDP]
                src_port = int(udp_layer.sport)
                dst_port = int(udp_layer.dport)
                proto_name = "UDP"

            elif packet.haslayer(ICMP):
                proto_name = "ICMP"
                src_port = 0
                dst_port = 0

            # 3. Construct 5-tuple
            flow_key = FlowKey(
                src_ip=src_ip,
                src_port=src_port,
                dst_ip=dst_ip,
                dst_port=dst_port,
                protocol=proto_name,
            )
            if bidirectional:
                flow_key = flow_key.to_bidirectional()

            # 4. Extract Timestamp and Wire Length
            pkt_time = float(packet.time) if hasattr(packet, "time") and packet.time else time.time()
            pkt_len = len(packet)

            return PacketRecord(
                timestamp=pkt_time,
                length=pkt_len,
                flow_key=flow_key,
                is_syn=is_syn,
                is_rst=is_rst,
            )

        except Exception as e:
            logger.debug("Failed to parse packet: %s", e)
            return None


# ============================================================================
# Multi-threaded Workers
# ============================================================================

class PacketCaptureWorker(threading.Thread):
    """
    Producer Thread: Captures live packets from a network interface using Scapy
    and pushes lightweight PacketRecord instances into a thread-safe Queue.
    """

    def __init__(
        self,
        interface: Optional[str],
        packet_queue: queue.Queue[PacketRecord],
        shutdown_event: threading.Event,
        bpf_filter: Optional[str] = None,
        bidirectional: bool = False,
    ):
        super().__init__(name="PacketCaptureWorker", daemon=True)
        self.interface = interface
        self.packet_queue = packet_queue
        self.shutdown_event = shutdown_event
        self.bpf_filter = bpf_filter
        self.bidirectional = bidirectional
        self.captured_count = 0
        self.dropped_count = 0
        self.parse_error_count = 0

    def _packet_callback(self, raw_packet):
        """Processes each captured packet and queues it."""
        if self.shutdown_event.is_set():
            return

        record = PacketParser.parse(raw_packet, bidirectional=self.bidirectional)
        if record is None:
            self.parse_error_count += 1
            return

        self.captured_count += 1
        try:
            # Non-blocking put to avoid stalling the sniffer if consumer slows down
            self.packet_queue.put_nowait(record)
        except queue.Full:
            self.dropped_count += 1
            if self.dropped_count % 1000 == 1:
                logger.warning(
                    "Packet queue full! Dropped %d packets due to backpressure.",
                    self.dropped_count,
                )

    def run(self):
        logger.info(
            "Starting live packet capture on interface '%s' (BPF filter: '%s')...",
            self.interface or "default",
            self.bpf_filter or "none",
        )

        if scapy is None:
            logger.error("Scapy library is not installed! Cannot capture packets.")
            return

        # Continuous sniffing loop with periodic timeout to check shutdown_event
        while not self.shutdown_event.is_set():
            try:
                scapy.sniff(
                    iface=self.interface,
                    filter=self.bpf_filter,
                    prn=self._packet_callback,
                    store=False,
                    timeout=1.0,  # Periodic timeout allows checking shutdown_event
                    stop_filter=lambda _: self.shutdown_event.is_set(),
                )
            except PermissionError as pe:
                logger.critical(
                    "Permission denied capturing network packets: %s. "
                    "Please run this script with Administrative/Root privileges.",
                    pe,
                )
                self.shutdown_event.set()
                break
            except RuntimeError as re:
                err_msg = str(re).lower()
                if "winpcap" in err_msg or "libpcap" in err_msg or "pcap" in err_msg:
                    logger.critical(
                        "Packet capture driver not found! (%s)\n"
                        "--> Windows users: Install Npcap (https://npcap.com/) with 'WinPcap API-compatible mode' enabled.\n"
                        "--> Linux users: Ensure libpcap is installed and run with sudo.",
                        re,
                    )
                else:
                    logger.critical("Runtime error during capture: %s", re)
                self.shutdown_event.set()
                break
            except Exception as e:
                if self.shutdown_event.is_set():
                    break
                logger.error("Unexpected error in packet capture loop: %s", e)
                time.sleep(0.5)

        logger.info(
            "PacketCaptureWorker terminated cleanly. Total captured: %d, dropped: %d",
            self.captured_count,
            self.dropped_count,
        )


class SlidingWindowAggregator(threading.Thread):
    """
    Consumer Thread: Pulls PacketRecords from the queue, maintains per-flow deques,
    prunes expired packets exceeding the sliding window duration, and computes
    statistical flow metrics on every slide interval.
    """

    def __init__(
        self,
        packet_queue: queue.Queue[PacketRecord],
        shutdown_event: threading.Event,
        window_size: float = 5.0,
        slide_interval: float = 1.0,
        metrics_callback: Optional[Callable[[List[FlowMetrics]], None]] = None,
        use_packet_time: bool = False,
    ):
        super().__init__(name="SlidingWindowAggregator", daemon=True)
        self.packet_queue = packet_queue
        self.shutdown_event = shutdown_event
        self.window_size = float(window_size)
        self.slide_interval = float(slide_interval)
        self.metrics_callback = metrics_callback
        self.use_packet_time = use_packet_time

        # Per-flow packet buffer: flow_key -> deque of PacketRecord
        self.flow_buffers: Dict[FlowKey, Deque[PacketRecord]] = defaultdict(deque)
        self.last_slide_time = time.time()
        self.latest_packet_time = 0.0
        self.total_flows_processed = 0

    def _ingest_packet(self, record: PacketRecord):
        """Adds a packet to the appropriate flow deque."""
        self.flow_buffers[record.flow_key].append(record)
        if record.timestamp > self.latest_packet_time:
            self.latest_packet_time = record.timestamp

    def _drain_queue(self, timeout: float = 0.1):
        """Drains pending packets from the queue."""
        try:
            record = self.packet_queue.get(timeout=timeout)
            self._ingest_packet(record)
            self.packet_queue.task_done()

            # Batch drain remaining items without blocking
            while not self.packet_queue.empty():
                record = self.packet_queue.get_nowait()
                self._ingest_packet(record)
                self.packet_queue.task_done()
        except queue.Empty:
            pass

    def evaluate_window(self, current_time: Optional[float] = None) -> List[FlowMetrics]:
        """
        Prunes packets outside the [window_start, window_end] interval and
        extracts flow metrics for all currently active flows.
        """
        if current_time is not None:
            window_end = float(current_time)
        elif self.use_packet_time and self.latest_packet_time > 0:
            window_end = self.latest_packet_time
        else:
            window_end = time.time()

        window_start = window_end - self.window_size

        metrics_list: List[FlowMetrics] = []
        expired_flows: List[FlowKey] = []

        for flow_key, pkt_deque in self.flow_buffers.items():
            # 1. Prune expired packets outside the sliding window
            while pkt_deque and pkt_deque[0].timestamp < window_start:
                pkt_deque.popleft()

            # 2. If no packets remain in the current window, mark flow for removal
            if not pkt_deque:
                expired_flows.append(flow_key)
                continue

            # 3. Calculate metrics over active packets in this window
            packet_count = len(pkt_deque)
            byte_count = sum(p.length for p in pkt_deque)

            first_time = pkt_deque[0].timestamp
            last_time = pkt_deque[-1].timestamp
            flow_duration = max(0.0, last_time - first_time)

            # Safeguard division by zero when duration is 0 (single packet or sub-microsecond)
            if flow_duration > 0.0:
                bytes_per_sec = byte_count / flow_duration
                packets_per_sec = packet_count / flow_duration
            else:
                bytes_per_sec = 0.0
                packets_per_sec = 0.0

            # TCP flag ratios
            syn_count = sum(1 for p in pkt_deque if p.is_syn)
            rst_count = sum(1 for p in pkt_deque if p.is_rst)
            syn_flag_ratio = (syn_count / packet_count) if packet_count > 0 else 0.0
            tcp_rst_ratio = (rst_count / packet_count) if packet_count > 0 else 0.0

            metrics = FlowMetrics(
                flow_key=flow_key,
                window_start=window_start,
                window_end=window_end,
                packet_count=packet_count,
                byte_count=byte_count,
                flow_duration=flow_duration,
                bytes_per_sec=bytes_per_sec,
                packets_per_sec=packets_per_sec,
                syn_flag_ratio=syn_flag_ratio,
                tcp_rst_ratio=tcp_rst_ratio,
            )
            metrics_list.append(metrics)

        # 4. Clean up inactive flows to prevent memory leaks
        for flow_key in expired_flows:
            del self.flow_buffers[flow_key]

        self.total_flows_processed += len(metrics_list)
        return metrics_list

    def run(self):
        logger.info(
            "Starting SlidingWindowAggregator (window_size=%.1fs, slide_interval=%.1fs)...",
            self.window_size,
            self.slide_interval,
        )

        while not self.shutdown_event.is_set():
            self._drain_queue(timeout=0.1)
            now = time.time()
            if (now - self.last_slide_time) >= self.slide_interval:
                eval_time = self.latest_packet_time if self.use_packet_time else now
                metrics = self.evaluate_window(eval_time)
                if self.metrics_callback and metrics:
                    try:
                        self.metrics_callback(metrics)
                    except Exception as e:
                        logger.error("Error in metrics callback: %s", e)
                self.last_slide_time = now

        # Flush remaining queue items on shutdown
        self._drain_queue(timeout=0.2)
        final_time = self.latest_packet_time if self.use_packet_time else time.time()
        final_metrics = self.evaluate_window(final_time)
        if self.metrics_callback and final_metrics:
            try:
                self.metrics_callback(final_metrics)
            except Exception as e:
                logger.error("Error in final metrics callback: %s", e)

        logger.info("SlidingWindowAggregator terminated cleanly.")


# ============================================================================
# Output Exporters (Console, CSV, JSON)
# ============================================================================

class FlowExporter:
    """
    Handles formatting and exporting flow metrics to terminal, CSV, or JSON.
    """

    def __init__(
        self,
        console_output: bool = True,
        csv_path: Optional[str] = None,
        json_path: Optional[str] = None,
    ):
        self.console_output = console_output
        self.csv_path = csv_path
        self.json_path = json_path
        self._csv_file = None
        self._csv_writer = None
        self._json_file = None

        if self.csv_path:
            self._init_csv()
        if self.json_path:
            self._init_json()

    def _init_csv(self):
        file_exists = os.path.exists(self.csv_path)
        self._csv_file = open(self.csv_path, mode="a", newline="", encoding="utf-8")
        fieldnames = [
            "timestamp",
            "window_start",
            "window_end",
            "src_ip",
            "src_port",
            "dst_ip",
            "dst_port",
            "protocol",
            "packet_count",
            "byte_count",
            "flow_duration",
            "bytes_per_sec",
            "packets_per_sec",
            "syn_flag_ratio",
            "tcp_rst_ratio",
        ]
        self._csv_writer = csv.DictWriter(self._csv_file, fieldnames=fieldnames)
        if not file_exists:
            self._csv_writer.writeheader()
            self._csv_file.flush()

    def _init_json(self):
        self._json_file = open(self.json_path, mode="a", encoding="utf-8")

    def export(self, metrics_list: List[FlowMetrics]):
        """Exports a batch of flow metrics."""
        if not metrics_list:
            return

        now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # 1. Console Table Output
        if self.console_output:
            print("\n" + "=" * 115)
            print(
                f"[{now_str}] 5s Sliding Window | Active Flows: {len(metrics_list)}"
            )
            print("-" * 115)
            header = (
                f"{'Source':<23} {'Destination':<23} {'Proto':<6} "
                f"{'Dur(s)':<8} {'Pkts':<6} {'Bytes':<9} {'Bytes/s':<11} "
                f"{'Pkts/s':<9} {'SYN%':<7} {'RST%':<7}"
            )
            print(header)
            print("-" * 115)

            for m in metrics_list:
                src = f"{m.flow_key.src_ip}:{m.flow_key.src_port}"
                dst = f"{m.flow_key.dst_ip}:{m.flow_key.dst_port}"
                print(
                    f"{src:<23} -> {dst:<20} {m.flow_key.protocol:<6} "
                    f"{m.flow_duration:<8.4f} {m.packet_count:<6} {m.byte_count:<9} "
                    f"{m.bytes_per_sec:<11.2f} {m.packets_per_sec:<9.2f} "
                    f"{m.syn_flag_ratio * 100:>5.1f}%  {m.tcp_rst_ratio * 100:>5.1f}%"
                )

        # 2. CSV File Export
        if self._csv_writer and self._csv_file:
            for m in metrics_list:
                row = m.to_dict()
                row["timestamp"] = now_str
                self._csv_writer.writerow(row)
            self._csv_file.flush()

        # 3. JSON Lines Export
        if self._json_file:
            for m in metrics_list:
                data = m.to_dict()
                data["timestamp"] = now_str
                self._json_file.write(json.dumps(data) + "\n")
            self._json_file.flush()

    def close(self):
        """Closes open file descriptors."""
        if self._csv_file:
            self._csv_file.close()
        if self._json_file:
            self._json_file.close()


# ============================================================================
# Main Controller / Pipeline Manager
# ============================================================================

class FlowMonitor:
    """
    Coordinates packet capture and sliding window flow analysis threads.
    Handles graceful initialization and shutdown.
    """

    def __init__(
        self,
        interface: Optional[str] = None,
        bpf_filter: Optional[str] = None,
        window_size: float = 5.0,
        slide_interval: float = 1.0,
        bidirectional: bool = False,
        exporter: Optional[FlowExporter] = None,
        queue_maxsize: int = 20000,
    ):
        self.interface = interface
        self.bpf_filter = bpf_filter
        self.window_size = window_size
        self.slide_interval = slide_interval
        self.bidirectional = bidirectional
        self.exporter = exporter or FlowExporter()

        self.shutdown_event = threading.Event()
        self.packet_queue: queue.Queue[PacketRecord] = queue.Queue(maxsize=queue_maxsize)

        self.capture_worker: Optional[PacketCaptureWorker] = None
        self.aggregator_worker: Optional[SlidingWindowAggregator] = None

    def start(self):
        """Starts worker threads and registers signal handlers."""
        logger.info("Initializing Network Flow Monitor...")

        self.aggregator_worker = SlidingWindowAggregator(
            packet_queue=self.packet_queue,
            shutdown_event=self.shutdown_event,
            window_size=self.window_size,
            slide_interval=self.slide_interval,
            metrics_callback=self.exporter.export,
        )

        self.capture_worker = PacketCaptureWorker(
            interface=self.interface,
            packet_queue=self.packet_queue,
            shutdown_event=self.shutdown_event,
            bpf_filter=self.bpf_filter,
            bidirectional=self.bidirectional,
        )

        self.aggregator_worker.start()
        self.capture_worker.start()

        # Register OS signal handlers for clean SIGINT / SIGTERM shutdown
        def _signal_handler(sig, frame):
            logger.info("Shutdown signal received (%s). Stopping gracefully...", sig)
            self.stop()

        try:
            signal.signal(signal.SIGINT, _signal_handler)
            signal.signal(signal.SIGTERM, _signal_handler)
        except (ValueError, AttributeError):
            # Might fail in non-main threads or specific Windows environments
            pass

    def stop(self):
        """Signals all worker threads to terminate and waits for completion."""
        if self.shutdown_event.is_set():
            return
        logger.info("Signaling workers to stop...")
        self.shutdown_event.set()

        if self.capture_worker and self.capture_worker.is_alive():
            self.capture_worker.join(timeout=3.0)

        if self.aggregator_worker and self.aggregator_worker.is_alive():
            self.aggregator_worker.join(timeout=3.0)

        self.exporter.close()
        logger.info("Network Flow Monitor stopped successfully.")

    def run_forever(self):
        """Blocks the main thread while monitoring workers."""
        self.start()
        try:
            while not self.shutdown_event.is_set():
                time.sleep(0.5)
        except KeyboardInterrupt:
            logger.info("Ctrl+C intercepted by user.")
        finally:
            self.stop()


# ============================================================================
# Synthetic Replay / Verification Mode
# ============================================================================

def run_synthetic_test(window_size: float = 5.0, slide_interval: float = 1.0):
    """
    Generates synthetic packets to verify multi-threaded sliding window
    aggregation and metrics extraction without requiring live capture drivers.
    """
    print("\n" + "=" * 80)
    print("RUNNING SYNTHETIC FLOW TEST (Zero-Dependency Self-Verification)")
    print("=" * 80)

    test_queue: queue.Queue[PacketRecord] = queue.Queue()
    shutdown_event = threading.Event()
    exporter = FlowExporter(console_output=True)

    aggregator = SlidingWindowAggregator(
        packet_queue=test_queue,
        shutdown_event=shutdown_event,
        window_size=window_size,
        slide_interval=slide_interval,
        metrics_callback=None,  # Handled manually in test to display step-by-step
        use_packet_time=True,
    )
    aggregator.start()

    # Create synthetic flows:
    # Flow 1: Normal HTTP session (SYN -> ACK -> Data -> Data -> FIN)
    # Flow 2: TCP SYN Flood attempt (all SYN flags)
    # Flow 3: Port Scan / Reset flow (RST flags)
    # Flow 4: UDP streaming flow
    base_time = 1000.0

    synthetic_packets = [
        # Flow 1 (Normal TCP): 5 packets, 3046 bytes, duration 3.0s, 1 SYN
        PacketRecord(base_time + 0.1, 74, FlowKey("192.168.1.10", 54321, "93.184.216.34", 80, "TCP"), is_syn=True, is_rst=False),
        PacketRecord(base_time + 0.5, 66, FlowKey("192.168.1.10", 54321, "93.184.216.34", 80, "TCP"), is_syn=False, is_rst=False),
        PacketRecord(base_time + 1.2, 1420, FlowKey("192.168.1.10", 54321, "93.184.216.34", 80, "TCP"), is_syn=False, is_rst=False),
        PacketRecord(base_time + 2.4, 1420, FlowKey("192.168.1.10", 54321, "93.184.216.34", 80, "TCP"), is_syn=False, is_rst=False),
        PacketRecord(base_time + 3.1, 66, FlowKey("192.168.1.10", 54321, "93.184.216.34", 80, "TCP"), is_syn=False, is_rst=False),

        # Flow 2 (SYN Scan / Flood): 4 SYN packets, 240 bytes, duration 2.0s, 100% SYN
        PacketRecord(base_time + 0.2, 60, FlowKey("10.0.0.5", 40001, "192.168.1.1", 22, "TCP"), is_syn=True, is_rst=False),
        PacketRecord(base_time + 0.8, 60, FlowKey("10.0.0.5", 40001, "192.168.1.1", 22, "TCP"), is_syn=True, is_rst=False),
        PacketRecord(base_time + 1.5, 60, FlowKey("10.0.0.5", 40001, "192.168.1.1", 22, "TCP"), is_syn=True, is_rst=False),
        PacketRecord(base_time + 2.2, 60, FlowKey("10.0.0.5", 40001, "192.168.1.1", 22, "TCP"), is_syn=True, is_rst=False),

        # Flow 3 (RST Abort): 2 packets, 120 bytes, 1 RST packet (50% RST)
        PacketRecord(base_time + 1.0, 60, FlowKey("172.16.0.2", 50100, "172.16.0.1", 443, "TCP"), is_syn=False, is_rst=False),
        PacketRecord(base_time + 1.8, 60, FlowKey("172.16.0.2", 50100, "172.16.0.1", 443, "TCP"), is_syn=False, is_rst=True),

        # Flow 4 (UDP DNS): 1 packet, 78 bytes, duration 0s
        PacketRecord(base_time + 2.0, 78, FlowKey("192.168.1.10", 51234, "8.8.8.8", 53, "UDP"), is_syn=False, is_rst=False),
    ]

    print(f"Injecting {len(synthetic_packets)} synthetic packets into queue...")
    for pkt in synthetic_packets:
        test_queue.put(pkt)

    time.sleep(0.3)

    # 1. Evaluate Window at T = base_time + 4.0s (Window spans [T-1.0s, T+4.0s])
    print("\n--- Window Evaluation #1: Window [T-1.0s, T+4.0s] (All 4 Flows Active) ---")
    metrics_at_4s = aggregator.evaluate_window(current_time=base_time + 4.0)
    exporter.export(metrics_at_4s)

    # 2. Advance time to T = base_time + 7.5s (Window spans [T+2.5s, T+7.5s])
    # Packets before 2.5s are expired and pruned.
    # Flow 1 has 1 packet at 3.1s (still active in window).
    # Flows 2, 3, 4 only had packets <= 2.2s, so they are cleanly evicted.
    print("\n--- Window Evaluation #2: Window [T+2.5s, T+7.5s] (Sliding Eviction Verification) ---")
    metrics_at_7_5s = aggregator.evaluate_window(current_time=base_time + 7.5)
    exporter.export(metrics_at_7_5s)

    shutdown_event.set()
    aggregator.join(timeout=2.0)
    print("\nSynthetic test completed successfully!")


# ============================================================================
# CLI Interface
# ============================================================================

def list_network_interfaces():
    """Prints all detected network interfaces available to Scapy."""
    print("Available Network Interfaces:")
    print("-" * 60)
    if scapy is None:
        print("Scapy is not installed. Run: pip install scapy")
        return

    try:
        ifaces = scapy.get_if_list()
        for idx, iface in enumerate(ifaces):
            is_default = (iface == getattr(scapy.conf, "iface", None))
            marker = " (Default)" if is_default else ""
            print(f"[{idx:>2}] {iface}{marker}")
    except Exception as e:
        print(f"Failed to query network interfaces: {e}")


def main():
    parser = argparse.ArgumentParser(
        description="Modular Scapy-based Live Network Flow Aggregator with 5s Sliding Window",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "-i", "--interface",
        type=str,
        default=None,
        help="Network interface to capture from (default: Scapy default interface)",
    )
    parser.add_argument(
        "-b", "--bpf",
        type=str,
        default=None,
        help="BPF capture filter (e.g., 'tcp or udp', 'port 80')",
    )
    parser.add_argument(
        "-w", "--window",
        type=float,
        default=5.0,
        help="Sliding window duration in seconds",
    )
    parser.add_argument(
        "-s", "--step",
        type=float,
        default=1.0,
        help="Slide step / export interval in seconds",
    )
    parser.add_argument(
        "--bidirectional",
        action="store_true",
        help="Aggregate flows bidirectionally (A->B and B->A grouped together)",
    )
    parser.add_argument(
        "--csv",
        type=str,
        default=None,
        help="Path to CSV file for exporting flow records",
    )
    parser.add_argument(
        "--json",
        type=str,
        default=None,
        help="Path to JSON Lines file for exporting flow records",
    )
    parser.add_argument(
        "--list-interfaces",
        action="store_true",
        help="List available network interfaces and exit",
    )
    parser.add_argument(
        "--test-replay",
        action="store_true",
        help="Run self-verification test with synthetic traffic (no driver needed)",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable debug logging",
    )

    args = parser.parse_args()

    # Logging setup
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="[%(asctime)s] [%(levelname)s] [%(threadName)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    if args.list_interfaces:
        list_network_interfaces()
        return

    if args.test_replay:
        run_synthetic_test(window_size=args.window, slide_interval=args.step)
        return

    exporter = FlowExporter(
        console_output=True,
        csv_path=args.csv,
        json_path=args.json,
    )

    monitor = FlowMonitor(
        interface=args.interface,
        bpf_filter=args.bpf,
        window_size=args.window,
        slide_interval=args.step,
        bidirectional=args.bidirectional,
        exporter=exporter,
    )

    monitor.run_forever()


if __name__ == "__main__":
    main()
