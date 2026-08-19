#!/usr/bin/env python2.7
"""
Fast drift attack + data collection script for MiniCPS Digital Twin.
(exact code used, run inside the digital_twin Docker container)

Collects 900 rows per run:
  - First 540 rows (60%): NORMAL operation (attack=0)
  - Last  360 rows (40%): FAST DRIFT attack (attack=1)

Drift rate: 10% of sensor range per cycle (10x the slow drift rate)
  SENSOR1: 0.551 m/cycle  (10% of 5.51m range)
  SENSOR3: 0.09  m/cycle  (10% of 0.9m range)

Usage (inside container):
    python collect_fast_drift_proper.py <run_number>
    python collect_fast_drift_proper.py 1
"""

import sqlite3
import time
import csv
import os
import subprocess
from datetime import datetime as dt

DB = '/src/fp_db.sqlite'
LOG_DIR = '/src/logs'
NORMAL_ROWS = 540
ATTACK_ROWS = 360
TOTAL_ROWS = NORMAL_ROWS + ATTACK_ROWS


def read_val(name, pid):
    conn = sqlite3.connect(DB)
    cur = conn.cursor()
    cur.execute('SELECT value FROM fp_table WHERE name=? AND pid=?', (name, pid))
    row = cur.fetchone()
    conn.close()
    return float(row[0]) if row else 0.0


def write_val(name, pid, value):
    conn = sqlite3.connect(DB)
    cur = conn.cursor()
    cur.execute('UPDATE fp_table SET value=? WHERE name=? AND pid=?', (str(value), name, pid))
    conn.commit()
    conn.close()


import sys
run_id = int(sys.argv[1]) if len(sys.argv) > 1 else 1
OUTPUT = os.path.join(LOG_DIR, 'fast_run_{:03d}.csv'.format(run_id))

print('Collecting fast drift run {:03d}...'.format(run_id))
print('Output: ' + OUTPUT)

# Fast drift parameters — 10% of sensor range per cycle
SENSOR1_DRIFT = 0.551   # 10% of range per cycle
SENSOR3_DRIFT = 0.09    # 10% of range per cycle
SENSOR1_RESET = 5.81
SENSOR3_RESET = 0.9
SENSOR1_MIN   = 0.3
SENSOR3_MIN   = 0.0

fake_s1 = SENSOR1_RESET
fake_s3 = SENSOR3_RESET
attack_active = False

with open(OUTPUT, 'wb') as f:
    writer = csv.writer(f)
    writer.writerow([
        'run_id', 'timestamp', 'unix_time', 'cycle',
        'SENSOR1_ACTUAL', 'SENSOR1_SENSED',
        'SENSOR2_ACTUAL', 'SENSOR2_SENSED',
        'SENSOR3_ACTUAL', 'SENSOR3_SENSED',
        'ACTUATOR1', 'attack'
    ])

for cycle in range(TOTAL_ROWS):
    # Switch to attack mode after NORMAL_ROWS
    if cycle == NORMAL_ROWS:
        attack_active = True
        fake_s1 = read_val('SENSOR1-LL-tank', 1)
        fake_s3 = read_val('SENSOR3-LL-bottle', 3)
        print('Attack started at cycle ' + str(cycle))

    # Read actual (physical process) values
    s1_actual = read_val('SENSOR1-LL-tank', 1)
    s2_actual = read_val('SENSOR2-FL', 2)
    s3_actual = read_val('SENSOR3-LL-bottle', 3)
    act = read_val('ACTUATOR1-MV', 1)

    if attack_active:
        # Apply fast drift (10% of range per cycle)
        fake_s1 = round(fake_s1 - SENSOR1_DRIFT, 5)
        fake_s3 = round(fake_s3 - SENSOR3_DRIFT, 5)

        if fake_s1 <= SENSOR1_MIN:
            fake_s1 = SENSOR1_RESET
        if fake_s3 <= SENSOR3_MIN:
            fake_s3 = SENSOR3_RESET

        # Write fake (drifted) values to DB — overwrites what
        # the physical process wrote, so PLC reads the fake value
        write_val('SENSOR1-LL-tank', 1, fake_s1)
        write_val('SENSOR3-LL-bottle', 3, fake_s3)

        s1_sensed = fake_s1
        s3_sensed = fake_s3
        attack_label = 1
    else:
        s1_sensed = s1_actual
        s3_sensed = s3_actual
        attack_label = 0

    with open(OUTPUT, 'ab') as f:
        writer = csv.writer(f)
        writer.writerow([
            run_id, str(dt.now()), int(time.time()), cycle,
            s1_actual, s1_sensed,
            s2_actual, s2_actual,
            s3_actual, s3_sensed,
            int(act), attack_label
        ])

    if cycle % 100 == 0:
        print('Cycle {}/{}: S1_actual={:.3f} S1_sensed={:.3f} attack={}'.format(
            cycle, TOTAL_ROWS, s1_actual, s1_sensed, attack_label))

    time.sleep(1)

# Reset DB values to normal after attack completes
write_val('SENSOR1-LL-tank', 1, s1_actual)
write_val('SENSOR3-LL-bottle', 3, s3_actual)

print('Done! Saved {} rows to {}'.format(TOTAL_ROWS, OUTPUT))

# ═══════════════════════════════════════════════════════════════
# USED TO GENERATE:
#   fast_run_001.csv ... fast_run_100.csv
#   (100 files x 900 rows = 90,000 rows total)
#   Each file: 540 normal rows + 360 fast-drift attack rows
#
# Run in a loop for all 100 files:
#   for i in $(seq 1 100); do
#       docker exec digital_twin bash -c \
#           "cd /src && python collect_fast_drift_proper.py $i"
#   done
# ═══════════════════════════════════════════════════════════════
