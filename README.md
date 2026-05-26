# Securing ICS using Digital Twin
## Dissertation by Navaneeth
## Supervisor: Matthew Kuo

## Overview
This repository contains the complete implementation of
1% drift attacks on a MiniCPS ICS bottle filling plant
digital twin with cascade analysis and LLM detection.

## System Architecture
- Ubuntu 24 VirtualBox
- Docker container: digital_twin
- Mininet virtual network
- 5 nodes: PLC1, PLC2, PLC3, HMI, Attacker
- SQLite database: fp_db.sqlite (virtual wire)

## Network
- PLC1 (master):   10.0.0.1
- PLC2 (flow):     10.0.0.2
- PLC3 (bottle):   10.0.0.3
- HMI:             10.0.0.4
- Attacker:        10.0.0.5

## Sensors
- SENSOR1-LL-tank:    0.3m to 5.81m  (tank level)
- SENSOR2-FL:         0.0 or 2.45 m3/h (flow rate)
- SENSOR3-LL-bottle:  0.0m to 0.9m   (bottle level)
- ACTUATOR1-MV:       0=closed 1=open (valve)

## Attacks Implemented

### Attack 1 - SENSOR1 1% Drift Attack
File:    attacks/attack_slow_refill_1pct.py
Target:  SENSOR1-LL-tank
Drift:   0.0551m per cycle (1% of 5.51m range)
Effect:  Tank level reported lower than reality
Damage:  Pump protection disabled - pump runs dry
Run:     python attack_slow_refill_1pct.py SENSOR1 5

### Attack 2 - SENSOR3 1% Drift Attack
File:    attacks/attack_slow_refill_1pct.py
Target:  SENSOR3-LL-bottle
Drift:   0.009m per cycle (1% of 0.9m range)
Effect:  Bottle level reported wrong
Damage:  Bottles overflow - valve open 94% of time
Run:     python attack_slow_refill_1pct.py SENSOR3 5

## Results Summary

### SENSOR1 Attack
Normal tank avg:   3.22m (53%)
Attack tank avg:   2.78m (45%) -8%
Normal valve open: 77%
Attack valve open: 82% +5%
Physical damage:   Pump protection disabled

### SENSOR3 Attack
Normal bottle avg: 0.45m (50%)
Attack bottle avg: 0.44m (49%) -1%
Normal valve open: 77%
Attack valve open: 94% +17%
Physical damage:   Bottles overflow

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

### Step 4 - Run cascade monitor simultaneously
docker exec -it digital_twin bash -c "cd /src && python monitor_cascade_v3.py SENSOR1_1pct 5"

## Repository Structure
setup/                  core simulation files
attacks/                attack scripts
monitor/                cascade monitor
detector/               LLM hybrid detector
graphs/no_attack/       normal operation graphs
graphs/sensor1_attack/  SENSOR1 attack graphs
graphs/sensor3_attack/  SENSOR3 attack graphs
graphs/cascade_matrix/  cascade impact matrix
results/                CSV log files

## Reference
Based on DT-based-IDS-framework
https://github.com/sebavarghese/DT-based-IDS-framework

Paper reference:
Kampourakis et al. 2026
Systematic Integration of Digital Twins and
Constrained LLMs for Interpretable
Cyber-Physical Anomaly Detection
