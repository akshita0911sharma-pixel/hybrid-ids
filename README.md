# Modular Scapy Network Flow Aggregator (5-Second Sliding Window)

A high-performance, modular Python network flow telemetry and feature extraction engine designed for network monitoring, ML-based Intrusion Detection Systems (IDS), and traffic analysis.

It captures live packets using **Scapy**, groups them into **5-tuple network flows** `(Source IP, Source Port, Dest IP, Dest Port, Protocol)`, and continuously calculates statistical flow metrics over a **5-second sliding window**.

---

## Features Extracted per Flow

For every active flow within the sliding window, the aggregator computes:

| Feature | Type | Description |
| :--- | :--- | :--- |
| `flow_duration` | float (seconds) | Time elapsed between the first and last packet observed in the current window: $t_{\text{last}} - t_{\text{first}}$. |
| `bytes_per_sec` | float (B/s) | Flow throughput: $\frac{\sum \text{packet\_bytes}}{\text{flow\_duration}}$ (zero-division guarded). |
| `packets_per_sec` | float (pkts/s) | Flow packet rate: $\frac{N}{\text{flow\_duration}}$ (zero-division guarded). |
| `syn_flag_ratio` | float $[0.0, 1.0]$ | Proportion of packets in the flow having the TCP SYN flag set: $\frac{N_{\text{SYN}}}{N}$. |
| `tcp_rst_ratio` | float $[0.0, 1.0]$ | Proportion of packets in the flow having the TCP RST flag set: $\frac{N_{\text{RST}}}{N}$. |

*Auxiliary fields included:* `src_ip`, `src_port`, `dst_ip`, `dst_port`, `protocol`, `packet_count`, `byte_count`, `window_start`, and `window_end`.

---

## Architecture & Multi-Threading

The system decouples network sniffing from flow aggregation using a thread-safe producer-consumer model:

```
                      +---------------------------------------+
                      |       Raw Network Interface / PCAP    |
                      +---------------------------------------+
                                          |
                                          v
+-----------------------------------------------------------------------------------+
|  [Producer Thread] PacketCaptureWorker                                             |
|   - Scapy `sniff()` with non-blocking BPF filtering                               |
|   - Lightweight header parsing via `PacketParser` (IPv4, IPv6, TCP, UDP, ICMP)    |
|   - Bitmask TCP flag evaluation (SYN=0x02, RST=0x04)                              |
|   - Memory isolation: emits compact `PacketRecord` (avoids Scapy object bloat)    |
+-----------------------------------------------------------------------------------+
                                          |
                                          v  Thread-safe bounded queue.Queue
+-----------------------------------------------------------------------------------+
|  [Consumer Thread] SlidingWindowAggregator                                        |
|   - Fast batch queue draining into per-flow `collections.deque`                   |
|   - Sliding Window: Evicts expired packets older than `T_now - window_size` (5s)  |
|   - Prunes dead flows from memory to avoid memory leaks                           |
|   - Computes mathematical metrics (duration, rates, ratios) with zero-div guards  |
|   - Periodic timer slide (default: every 1.0s)                                    |
+-----------------------------------------------------------------------------------+
                                          |
                                          v
                      +---------------------------------------+
                      |         FlowExporter / Sinks          |
                      |   - Live Terminal Formatted Table     |
                      |   - CSV Logging (`--csv flows.csv`)   |
                      |   - JSONL Logging (`--json logs.json`) |
                      +---------------------------------------+
```

### Key Design & Reliability Features
- **Clean Shutdown**: Listens for `SIGINT` (Ctrl+C) and `SIGTERM` via `threading.Event`, allowing both threads to flush pending queues and terminate cleanly without zombie sockets.
- **Memory Safety**: Expired packets older than 5.0 seconds are dropped from flow deques; empty flow keys are removed from active dictionaries.
- **Backpressure Protection**: Bounded queue prevents memory exhaustion during micro-bursts of high packet rates.
- **Zero-Division Resilience**: Flows with 1 packet or sub-microsecond durations safely output `0.0` for rate metrics without raising exceptions.
- **Windows & Linux Compatibility**: Works across operating systems with clear error reporting when drivers/permissions are missing.

---

## Prerequisites & Installation

### 1. Python Dependencies
```bash
pip install scapy
```

### 2. Network Driver Requirements
- **Windows**: Install [Npcap](https://npcap.com/) (recommended: check *"Install Npcap in WinPcap API-compatible Mode"*). Run your shell as **Administrator**.
- **Linux / macOS**: Ensure `libpcap` is installed (`sudo apt install libpcap-dev`) and run the script with root privileges (`sudo python ...`).

---

## Usage

### 1. Zero-Dependency Self-Verification (Synthetic Test)
Test the sliding window engine, feature extractor, and multi-threaded pipeline instantly on any machine without needing Npcap or root access:
```bash
python flow_aggregator.py --test-replay
```

### 2. List Available Interfaces
Find the exact name or GUID of your network interfaces:
```bash
python flow_aggregator.py --list-interfaces
```

### 3. Live Capture with Default Interface
```bash
# Capture on default interface with 5s sliding window evaluated every 1s
python flow_aggregator.py
```

### 4. Capture on a Specific Interface with BPF Filter
```bash
# Capture only TCP or UDP traffic on a specific interface
python flow_aggregator.py -i "Ethernet" -b "tcp or udp"
```

### 5. Export to CSV and JSON Lines
```bash
python flow_aggregator.py -i "Wi-Fi" --csv flows.csv --json flows.jsonl
```

### 6. Bidirectional Flows
Group bidirectional conversations ($A \leftrightarrow B$) into a single canonical 5-tuple:
```bash
python flow_aggregator.py --bidirectional
```

### 7. Custom Window and Slide Interval
```bash
# 10-second window evaluated every 2 seconds
python flow_aggregator.py -w 10.0 -s 2.0
```

---

## Example Console Output

```
===================================================================================================================
[2026-10-19 11:52:00] 5s Sliding Window | Active Flows: 4
-------------------------------------------------------------------------------------------------------------------
Source                  Destination             Proto  Dur(s)   Pkts   Bytes     Bytes/s     Pkts/s    SYN%    RST%   
-------------------------------------------------------------------------------------------------------------------
192.168.1.10:54321      -> 93.184.216.34:80     TCP    3.0000   5      3046      1015.33     1.67       20.0%    0.0%
10.0.0.5:40001          -> 192.168.1.1:22       TCP    2.0000   4      240       120.00      2.00      100.0%    0.0%
172.16.0.2:50100        -> 172.16.0.1:443       TCP    0.8000   2      120       150.00      2.50        0.0%   50.0%
192.168.1.10:51234      -> 8.8.8.8:53           UDP    0.0000   1      78        0.00        0.00        0.0%    0.0%
```

---

## Running Automated Tests

Run the test suite verifying all 24 unit and integration test cases:
```bash
python -m unittest discover -s . -p "test_*.py"
```

---

# SignatureEngine (Sub-Millisecond Heuristic & Signature Detection)

[`signature_engine.py`](signature_engine.py) provides a high-speed signature evaluation module designed to inspect incoming 5-tuple flow dictionaries against customizable heuristic, statistical, and payload rules.

- **Immediate Alerting**: High-confidence matching flows generate structured `Alert` objects with prioritized severity (`LOW`, `MEDIUM`, `HIGH`, `CRITICAL`) and are dispatched via `alerts_queue` and optional callbacks.
- **Secondary Queue Routing**: Clean (non-matching) flows are automatically routed to `secondary_queue` for ML-based anomaly detection or deep packet inspection.
- **Latency Requirement**: Guarantees average evaluation latency $< 0.05\,\text{ms}$ and 99th percentile $< 0.2\,\text{ms}$ (far below the $2.0\,\text{ms}$ threshold).

### Built-in Detection Rules
1. **TCP SYN Flood** (`RULE-DOS-SYN-FLOOD`): `CRITICAL` — High SYN ratio ($\ge 0.80$) with high packet rate ($\ge 15\,\text{pps}$).
2. **Stealth SYN Probe** (`RULE-RECON-SYN-PROBE`): `MEDIUM` — Isolated single-packet probe to sensitive ports ($21, 22, 23, 80, 443, 3389$, etc.).
3. **Connection Reset Sweep** (`RULE-RECON-RST-SWEEP`): `LOW` — Short-lived teardowns with high RST ratio ($\ge 0.50$).
4. **Stateful Multi-Port Scan** (`RULE-RECON-RAPID-PORTSCAN`): `HIGH` — Source IP targeting $\ge 8$ distinct destination ports across flows within a 5-second sliding window.
5. **SQL Injection Payload** (`RULE-APP-SQLI-INJECTION`): `HIGH` — Matches `UNION SELECT`, `' OR '1'='1`, `xp_cmdshell`, etc.
6. **Command Execution / RCE** (`RULE-APP-RCE-SHELL`): `CRITICAL` — Matches `/bin/sh`, `/etc/passwd`, `cmd.exe /c`, `powershell -nop`.
7. **XSS & Path Traversal** (`RULE-APP-XSS-SCRIPT`, `RULE-APP-DIR-TRAVERSAL`): `MEDIUM` / `HIGH` — Matches `<script>`, `../../`, etc.

### Benchmark Run
```bash
python signature_engine.py
```
*Benchmark Results (10,000 flows):*
- **Average Latency per Flow:** ~0.030 ms (30 microseconds)
- **99th Percentile Latency:** ~0.118 ms (118 microseconds)
- **Status:** PASSED (< 2.0 ms requirement)

---

# NetworkAnomalyDetector (Isolation Forest + Diurnal Feature Engineering)

[`anomaly_detector.py`](anomaly_detector.py) provides machine-learning-based network anomaly detection using `scikit-learn`'s `IsolationForest`, specifically engineered to handle daily cyclical traffic patterns.

### Why Diurnal Feature Engineering Matters
Enterprise networks experience predictable 24-hour cycles: business hours (09:00 - 18:00) naturally have high throughput, while night hours (00:00 - 06:00) are quiet. Standard models looking at raw volume flag busy work hours as false positives and miss stealthy off-hour data exfiltration.

The engine resolves this with two complementary feature engineering methods:
1. **Cyclical Time Encoding**:
   $$\text{hour\_sin} = \sin\left(\frac{2\pi \cdot \text{hour}}{24.0}\right), \quad \text{hour\_cos} = \cos\left(\frac{2\pi \cdot \text{hour}}{24.0}\right)$$
   Preserves angular continuity across the midnight boundary ($23:59 \to 00:00$).
2. **Hourly Volume Z-Scores**:
   $$Z_{\text{volume}} = \frac{x_{\text{volume}} - \mu_{\text{hour}}}{\sigma_{\text{hour}} + \epsilon}$$
   Evaluates each flow's volume against the historical baseline for that specific hour:
   - A $50\,\text{MB}$ burst at $14:00$ (where $\mu = 48\,\text{MB}$) has $Z \approx +0.2$ $\to$ **NORMAL**.
   - A $50\,\text{MB}$ burst at $03:00$ (where $\mu = 0.5\,\text{MB}$) has $Z > +100.0$ $\to$ **CRITICAL ANOMALY**.

### Training & Real-Time Prediction API
```python
from anomaly_detector import NetworkAnomalyDetector

# 1. Training (Batch)
detector = NetworkAnomalyDetector(n_estimators=100, contamination=0.02)
detector.fit(historical_flows_df)
detector.save_model("isolation_forest_model.joblib")

# 2. Real-Time Prediction (< 0.5 ms per flow)
detector = NetworkAnomalyDetector.load_model("isolation_forest_model.joblib")
result = detector.predict_flow(flow_dict)

print(result.is_anomaly)             # True / False
print(result.anomaly_score)           # Normalized [0.0, 1.0]
print(result.severity)                # NORMAL, LOW, MEDIUM, HIGH, CRITICAL
print(result.zscores)                 # Hourly Z-scores for volume metrics
```

### Demonstration & Verification
```bash
python anomaly_detector.py --demo
```

---

# HybridIDSOrchestrator (Central Multi-Stage Orchestrator)

[`orchestrator.py`](orchestrator.py) is the central pipeline orchestrator connecting Scapy live flow capture, Stage 1 Signature Engine, Stage 2 ML Anomaly Engine, and Alert Aggregation.

### Pipeline Architecture & Queue Routing

```
               [Live Network Interface / Replay PCAP]
                                 │
                                 ▼
                     [SlidingWindowAggregator]
                                 │  5-tuple flows (5.0s window)
                                 ▼
                         [raw_flow_queue]
                                 │
                                 ▼
                     [Stage 1: SignatureEngine]
                      ├── MATCH ───────────────────┐
                      └── UNMATCHED (Clean)        │
                                │                  │
                                ▼                  │
                         [stage2_queue]            │
                                │                  │
                                ▼                  │
                     [Stage 2: AnomalyDetector]    │
                      ├── ANOMALY ─────────────────┤
                      └── NORMAL                   │
                                │                  │
                             [Ignore]              ▼
                                          [alert_ingress_queue]
                                                   │
                                                   ▼
                                          [AlertAggregator]
                                        (30s Source IP Dedup)
                                                   │
                                                   ▼
                                         [Structured JSON Stream]
```

### Alert Deduplication Logic (30-Second Source IP Window)
- **Immediate First Alert:** The first alert for a given Source IP is emitted immediately as a structured JSON record with `"dedup_metadata": {"is_first_in_window": true}`.
- **Suppression of Duplicates:** Subsequent alerts from the same Source IP within 30 seconds are suppressed to prevent alert fatigue.
- **Severity Escalation:** If a subsequent alert from the same Source IP carries higher severity (e.g. `LOW` $\to$ `CRITICAL`), an immediate `SEVERITY_ESCALATION` JSON event is emitted.
- **Aggregated Summary:** When the 30-second window expires, if multiple alerts occurred, an `AGGREGATED_SUMMARY` JSON event is emitted detailing total alerts, suppressed count, highest severity, and all triggered rules.

### Running the Orchestrator

#### 1. End-to-End Synthetic Pipeline Demonstration
Verifies the complete pipeline without requiring live root / Npcap access:
```bash
python orchestrator.py --demo
```

#### 2. Live Capture with Signature & Anomaly Detection
```bash
# Capture live on an interface and log alerts to alerts.jsonl
python orchestrator.py -i "Ethernet" -b "tcp or udp" -o alerts.jsonl
```

---

# Streamlit SOC Dashboard (`app.py`)

[`app.py`](app.py) provides an interactive Security Operations Center (SOC) web dashboard displaying real-time threat intelligence.

### Dashboard Capabilities
1. **Live Traffic Volume vs Diurnal Baseline Line Chart:** Dual-line comparison showing real-time throughput versus expected historical diurnal baseline.
2. **Active Alert Feed with Severity Badges:** Color-coded badges (`CRITICAL`, `HIGH`, `MEDIUM`, `LOW`) and collapsible technical inspection cards with flow metrics and detection parameters.
3. **Top Attacking Source IPs Table:** Aggregated breakdown of threat actors, alert counts, highest severity, attack types, and last active timestamp.
4. **Detection Breakdown Visualizations:** Donut chart of Stage 1 Signature vs Stage 2 ML Anomaly proportions, plus severity distribution bar charts.
5. **Real-Time Auto-Refresh:** Continuously monitors and updates from `alerts.jsonl` with configurable intervals.

### Launching the Dashboard
```bash
streamlit run app.py
```
Open [http://localhost:8501](http://localhost:8501) in your browser.

---

## Full Workspace Automated Test Suite

Run all **42 unit and integration tests** across the entire Hybrid IDS codebase:
```bash
python -m unittest discover -s . -p "test_*.py"
```
*Coverage includes: Scapy flow capture, sliding-window calculations, signature matching, stateful port scanning, cyclical time encoding, diurnal Z-scores, Isolation Forest ML inference, and orchestrator 30s alert deduplication.*




