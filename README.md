# Detecting Stealthy Sensor Drift Attacks in ICS Using Digital Twins and LLMs

---

## Overview

This project investigates the detection of fast and slow False Data Injection (FDI) sensor drift attacks in Industrial Control Systems (ICS) using a MiniCPS based Digital Twin (DT) and a fine tuned Large Language Model (LLM).

The Secure Water Treatment (SWaT) dataset was first analysed to examine fast sensor drift behaviour. This behaviour was then used as a baseline for controlled fast drift experiments in MiniCPS. Slower drift configurations were subsequently introduced to investigate more gradual manipulation and its effects on the wider control process.

The project also examines whether attack behaviour learned at one drift rate generalises to another. Qwen2.5 0.5B Instruct was adapted using Low Rank Adaptation (LoRA) for attack classification, while Retrieval Augmented Generation (RAG) was used after detection to provide additional process and cybersecurity context.

---

## System Architecture

The experimental environment is based on a MiniCPS water filling Digital Twin adapted from the DT based IDS framework.

### Environment

- Ubuntu 24 VirtualBox VM
- Docker container: `digital_twin`
- MiniCPS and Mininet virtual environment
- SQLite process database: `fp_db.sqlite`
- Virtual PLCs, HMI, sensors, actuator, and attacker node
- CSV logging for experimental process data
- EtherNet/IP based communication

### Virtual Network Nodes

| Component | Role | IP Address |
|---|---|---|
| PLC1 | Main controller | `10.0.0.1` |
| PLC2 | Flow monitoring PLC | `10.0.0.2` |
| PLC3 | Bottle level monitoring PLC | `10.0.0.3` |
| HMI | Human Machine Interface | `10.0.0.4` |
| Attacker | Internal attack node | `10.0.0.5` |

---

## Process Variables

| Variable | Description | Range / State |
|---|---|---|
| `SENSOR1` | Tank level | `0.30 m` to `5.81 m` |
| `SENSOR2` | Flow rate | `0` or `2.45` |
| `SENSOR3` | Bottle level | `0.00 m` to `0.90 m` |
| `ACTUATOR1` | Motor valve | `0 = closed`, `1 = open` |

The experimental environment maintains separate `ACTUAL` and `SENSED` measurements.

- `ACTUAL` represents the underlying simulated process state.
- `SENSED` represents the measurement available to the controller.
- During normal operation, `ACTUAL = SENSED`.
- During an attack, the attack script modifies the `SENSED` measurement while retaining `ACTUAL` for analysis and validation.

---

## Sensor Drift Attacks

Four primary MiniCPS attack scenarios were evaluated:

| Target | Fast Drift | Slow Drift |
|---|---:|---:|
| SENSOR1 | 10% of operating span per cycle | 1% of operating span per cycle |
| SENSOR3 | 10% of operating span per cycle | 1% of operating span per cycle |

### SENSOR1

Operating span:

`5.81 - 0.30 = 5.51 m`

Fast drift:

`0.551 m/cycle`

Slow drift:

`0.0551 m/cycle`

### SENSOR3

Operating span:

`0.90 - 0.00 = 0.90 m`

Fast drift:

`0.09 m/cycle`

Slow drift:

`0.009 m/cycle`

The attack scripts incrementally modify the controller visible `SENSED` measurement through the shared SQLite database. When the manipulated value reaches the configured sensor bound, it is reset to the opposite bound, producing a repeating sawtooth drift pattern during the active attack period.

---

## Preliminary Slow Drift Testing

Before selecting the final 1% slow drift configuration, preliminary SENSOR1 experiments were performed using smaller drift rates.

- `0.005 m/cycle`, approximately `0.09%` of the SENSOR1 operating span
- `0.5%` of the SENSOR1 operating span
- Final configuration: `1%`

The smaller configurations produced very gradual divergence within the experimental duration. The 1% rate was therefore selected to provide a clearer gradual manipulation while remaining substantially below the 10% fast drift configuration.

---

## Cascading Process Effects

The experiments showed that the impact of sensor manipulation was not limited to the attacked sensor.

### SENSOR1 Slow Drift

- Mean SENSOR1 level: `3.22 m → 2.78 m`
- Mean valve state: `0.77 → 0.82`
- The underlying tank level fell below the configured `0.30 m` lower limit while the controller visible measurement did not reflect the same condition.
- ACTUATOR1 remained open during part of this period.

### SENSOR3 Slow Drift

- Mean SENSOR3 level: `0.45 m → 0.44 m`
- Mean valve state: `0.77 → 0.94`
- The underlying bottle level exceeded the configured `0.90 m` upper limit while the controller visible measurement remained lower.
- ACTUATOR1 remained open during part of the overflow condition.

These results show that gradual sensor manipulation can become visible through changes in related sensors and actuator behaviour even when the average value of the attacked sensor remains close to normal.

---

## Dataset Generation

Separate process datasets were generated for the fast and slow drift experiments.

Each row contains:

- `ACTUAL` sensor measurements
- `SENSED` sensor measurements
- actuator state
- process cycle information
- ground truth attack label

Ground truth labels were determined from the known attack injection periods:

- `0 = Normal`
- `1 = Attack`

The `ACTUAL` measurements and ground truth labels were retained for analysis and evaluation but were not supplied to the LLM as classification input.

---

## LLM Detection Pipeline

The generated process data were converted into overlapping observation windows.

- Window size: `30 seconds`
- Stride: `10 seconds`
- Overlap: `20 seconds`

For each sensor, the following features were calculated:

- first value
- last value
- change
- minimum
- maximum
- mean
- range

For ACTUATOR1:

- open ratio
- transition count

These features were converted into structured textual prompts and provided to:

**Qwen2.5 0.5B Instruct + LoRA**

LoRA configuration:

- Rank: `8`
- Alpha: `16`
- Dropout: `0.05`

The model produced one classification for each observation window:

```json
{"prediction":"Attack"}
