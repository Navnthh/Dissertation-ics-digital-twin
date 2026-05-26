# Securing ICS Using Digital Twin and LLM

**Dissertation by:** Navaneeth Rajendran  
**Supervisors:** Dr. Farzana Zahid, Dr. Matthew Kuo  

---

## Overview

---

## System Architecture

The experiment is based on a Digital Twin ICS testbed.

### Environment

- Ubuntu 24 VirtualBox VM
- Docker container: `digital_twin`
- Mininet virtual network
- SQLite database: `fp_db.sqlite`
- Digital Twin framework: `DT-based-IDS-framework`

### Virtual Network Nodes

| Component | Role | IP Address |
|---|---|---|
| PLC1 | Master PLC | `10.0.0.1` |
| PLC2 | Flow PLC | `10.0.0.2` |
| PLC3 | Bottle PLC | `10.0.0.3` |
| HMI | Human-Machine Interface | `10.0.0.4` |
| Attacker | Attack node | `10.0.0.5` |

---

## Sensors and Actuator

| Tag | Description | Range / Values |
|---|---|---|
| `SENSOR1-LL-tank` | Tank level sensor | `0.3m` to `5.81m` |
| `SENSOR2-FL` | Flow rate sensor | `0.0` or `2.45 m³/h` |
| `SENSOR3-LL-bottle` | Bottle level sensor | `0.0m` to `0.9m` |
| `ACTUATOR1-MV` | Valve actuator | `0 = closed`, `1 = open` |

---

## Attacks Implemented

### Attack 1: SENSOR1 1% Drift Attack

| Field | Description |
|---|---|
| File | `attacks/attack_slow_refill_1pct.py` |
| Target | `SENSOR1-LL-tank` |
| Drift | `0.0551m` per cycle |
| Drift Basis | 1% of `5.51m` tank range |
| Effect | Tank level is reported lower than reality |
| Physical Impact | Pump protection may be disabled, causing the pump to run dry |
| Run Command | `python attack_slow_refill_1pct.py SENSOR1 5` |

### Attack 2: SENSOR3 1% Drift Attack

| Field | Description |
|---|---|
| File | `attacks/attack_slow_refill_1pct.py` |
| Target | `SENSOR3-LL-bottle` |
| Drift | `0.009m` per cycle |
| Drift Basis | 1% of `0.9m` bottle range |
| Effect | Bottle level is reported incorrectly |
| Physical Impact | Bottles may overflow because the valve remains open longer |
| Run Command | `python attack_slow_refill_1pct.py SENSOR3 5` |

---

## Results Summary

### SENSOR1 Attack Results

| Metric | Normal Operation | Attack Scenario | Change |
|---|---:|---:|---:|
| Tank average level | `3.22m` / `53%` | `2.78m` / `45%` | `-8%` |
| Valve open time | `77%` | `82%` | `+5%` |
| Physical impact | Normal pump protection | Pump protection disabled | Pump may run dry |

### SENSOR3 Attack Results

| Metric | Normal Operation | Attack Scenario | Change |
|---|---:|---:|---:|
| Bottle average level | `0.45m` / `50%` | `0.44m` / `49%` | `-1%` |
| Valve open time | `77%` | `94%` | `+17%` |
| Physical impact | Normal filling behaviour | Valve remains open longer | Bottles may overflow |

---
## How to Run

### Step 1 - Start simulation
cd ~/DT-based-IDS-framework/DigitalTwin-SIEM-integration/deployments/docker
export HOST_IP=10.0.2.15
xhost +si:localuser:root
export DISPLAY=:0
docker-compose up -d
docker-compose exec digital_twin bash -lc "cd /src && mn -c && sleep 2 && python run.py"

### Step 2 - Trigger
docker-compose exec digital_twin bash -lc "cd /src && touch trigger.txt"

### Step 3 - Run attack
docker exec -it digital_twin bash
cd /src
python attack_slow_refill_1pct.py SENSOR1 5
cd /src
python attack_slow_refill_1pct.py SENSOR3 5


### Step 4 - Run cascade monitor simultaneously
docker exec -it digital_twin bash -c "cd /src && python monitor_cascade_v3.py SENSOR1_1pct 5"

## Reference
Based on DT-based-IDS-framework
https://github.com/sebavarghese/DT-based-IDS-framework

Paper reference:
