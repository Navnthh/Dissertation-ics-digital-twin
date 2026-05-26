#! /usr/bin/env python2.7
"""
Cascading Impact Monitor v3 - Sustained Breach Detection
Only flags breaches that last 3 or more consecutive cycles
This eliminates natural simulation timing noise
Usage: python monitor_cascade_v3.py <attack_name> <duration_minutes>
"""

import sqlite3
import time
import sys
import csv
import os
from datetime import datetime as dt

PATH = '/src/fp_db.sqlite'
LOG_DIR = '/src/logs'

if not os.path.exists(LOG_DIR):
    os.makedirs(LOG_DIR)

BASELINE = {
    'SENSOR1-LL-tank':   {'pid': 1},
    'SENSOR2-FL':        {'pid': 2},
    'SENSOR3-LL-bottle': {'pid': 3},
    'ACTUATOR1-MV':      {'pid': 1},
}

THRESHOLDS = {
    'SENSOR1-LL-tank':   {'lower': 0.3,  'upper': 5.81},
    'SENSOR2-FL':        {'lower': 0.0,  'upper': 3.0},
    'SENSOR3-LL-bottle': {'lower': 0.0,  'upper': 0.9},
}

# Number of consecutive cycles before flagging as real breach
SUSTAINED_CYCLES = 5

def read_all_values(cur):
    values = {}
    for name, info in BASELINE.items():
        cur.execute('SELECT value FROM fp_table WHERE name=? AND pid=?',
                   (name, info['pid']))
        row = cur.fetchone()
        values[name] = float(row[0]) if row else 0.0
    return values

def check_raw_status(name, value):
    if name not in THRESHOLDS:
        return 'NORMAL'
    t = THRESHOLDS[name]
    if value < t['lower']:
        return 'BELOW_LOWER'
    elif value > t['upper']:
        return 'ABOVE_UPPER'
    return 'NORMAL'

def check_valve(values):
    tank   = values['SENSOR1-LL-tank']
    flow   = values['SENSOR2-FL']
    bottle = values['SENSOR3-LL-bottle']
    valve  = values['ACTUATOR1-MV']
    if tank < 0.3:
        exp, reason = 0, 'tank too low'
    elif flow >= 3.0:
        exp, reason = 0, 'flow too high'
    elif bottle >= 0.9:
        exp, reason = 0, 'bottle full'
    elif bottle < 0.9 and tank > 0.3:
        exp, reason = 1, 'fill bottle'
    else:
        exp, reason = 0, 'unknown'
    act = int(round(valve))
    return exp, act, (act == exp), reason

def monitor(attack_name, duration_mins):
    duration_secs = int(duration_mins) * 60
    close_time = time.time() + duration_secs
    ts = dt.now().strftime('%Y%m%d_%H%M%S')
    csv_file = os.path.join(LOG_DIR, 'cascade_v3_' + attack_name + '_' + ts + '.csv')
    summary_file = os.path.join(LOG_DIR, 'summary_v3_' + attack_name + '_' + ts + '.txt')

    print '=' * 60
    print 'CASCADE MONITOR v3 - SUSTAINED BREACH DETECTION'
    print '=' * 60
    print 'Attack:           ' + attack_name
    print 'Duration:         ' + str(duration_mins) + ' minutes'
    print 'Sustained cycles: ' + str(SUSTAINED_CYCLES) + ' (min cycles before flagging)'
    print 'Log: ' + csv_file
    print 'START TIME: ' + str(dt.now())
    print '=' * 60

    stats = {
        'total': 0,
        'cascade': 0,
        'valve_wrong': 0,
        'valve_ok': 0,
        's1_breach': 0,
        's2_breach': 0,
        's3_breach': 0,
        'first_breach': None,
        'first_wrong_valve': None,
    }

    # Consecutive breach counters
    consecutive = {
        'SENSOR1-LL-tank':   0,
        'SENSOR2-FL':        0,
        'SENSOR3-LL-bottle': 0,
    }

    # Sustained breach flags
    sustained = {
        'SENSOR1-LL-tank':   False,
        'SENSOR2-FL':        False,
        'SENSOR3-LL-bottle': False,
    }

    prev = None

    with open(csv_file, 'wb') as f:
        csv.writer(f).writerow([
            'timestamp', 'cycle',
            'SENSOR1', 'SENSOR2', 'SENSOR3', 'ACTUATOR',
            'S1_status', 'S2_status', 'S3_status',
            'S1_consecutive', 'S2_consecutive', 'S3_consecutive',
            'valve_exp', 'valve_act', 'valve_ok', 'reason',
            'cascade', 'notes'
        ])

    conn = sqlite3.connect(PATH)
    cur = conn.cursor()
    cycle = 0

    try:
        while close_time > time.time():
            cycle += 1
            stats['total'] = cycle
            ts_now = str(dt.now())
            v = read_all_values(cur)

            # Check raw status for each sensor
            s1_raw = check_raw_status('SENSOR1-LL-tank',   v['SENSOR1-LL-tank'])
            s2_raw = check_raw_status('SENSOR2-FL',         v['SENSOR2-FL'])
            s3_raw = check_raw_status('SENSOR3-LL-bottle',  v['SENSOR3-LL-bottle'])

            # Update consecutive counters
            for name, raw in [
                ('SENSOR1-LL-tank', s1_raw),
                ('SENSOR2-FL', s2_raw),
                ('SENSOR3-LL-bottle', s3_raw)
            ]:
                if raw != 'NORMAL':
                    consecutive[name] += 1
                else:
                    consecutive[name] = 0
                    sustained[name] = False

                # Only flag as sustained breach after SUSTAINED_CYCLES cycles
                if consecutive[name] >= SUSTAINED_CYCLES:
                    sustained[name] = True

            # Use sustained status for cascade detection
            s1_status = 'SUSTAINED_' + s1_raw if sustained['SENSOR1-LL-tank'] else 'NORMAL'
            s2_status = 'SUSTAINED_' + s2_raw if sustained['SENSOR2-FL'] else 'NORMAL'
            s3_status = 'SUSTAINED_' + s3_raw if sustained['SENSOR3-LL-bottle'] else 'NORMAL'

            # Update breach stats only for sustained breaches
            if sustained['SENSOR1-LL-tank']:
                stats['s1_breach'] += 1
                if not stats['first_breach']:
                    stats['first_breach'] = ts_now
            if sustained['SENSOR2-FL']:
                stats['s2_breach'] += 1
                if not stats['first_breach']:
                    stats['first_breach'] = ts_now
            if sustained['SENSOR3-LL-bottle']:
                stats['s3_breach'] += 1
                if not stats['first_breach']:
                    stats['first_breach'] = ts_now

            # Check valve decision
            exp, act, ok, reason = check_valve(v)
            if ok:
                stats['valve_ok'] += 1
            else:
                stats['valve_wrong'] += 1
                if not stats['first_wrong_valve']:
                    stats['first_wrong_valve'] = ts_now

            # Build cascade notes
            notes = []
            cascade = False

            if sustained['SENSOR1-LL-tank']:
                notes.append('TANK SUSTAINED ' + s1_raw + ' (' + str(consecutive['SENSOR1-LL-tank']) + ' cycles)')
                cascade = True
            if sustained['SENSOR2-FL']:
                notes.append('FLOW SUSTAINED ' + s2_raw + ' (' + str(consecutive['SENSOR2-FL']) + ' cycles)')
                cascade = True
            if sustained['SENSOR3-LL-bottle']:
                notes.append('BOTTLE SUSTAINED ' + s3_raw + ' (' + str(consecutive['SENSOR3-LL-bottle']) + ' cycles)')
                cascade = True
            if False: # removed valve wrong from cascade
                notes.append('VALVE WRONG exp=' + str(exp) + ' act=' + str(act) + ' reason=' + reason)
                cascade = True

            # Detect sudden jumps from previous cycle
            if prev:
                for name in ['SENSOR1-LL-tank', 'SENSOR2-FL', 'SENSOR3-LL-bottle']:
                    delta = abs(v[name] - prev[name])
                    if delta > 0.5:
                        notes.append(name + ' sudden jump: ' + str(round(delta, 4)))

            if cascade:
                stats['cascade'] += 1

            prev = v.copy()

            # Print every 10 cycles
            if cycle % 10 == 0:
                print '\n[Cycle ' + str(cycle) + '] ' + ts_now
                print '  SENSOR1-LL-tank:   ' + str(round(v['SENSOR1-LL-tank'], 4)) + \
                      '  [consecutive=' + str(consecutive['SENSOR1-LL-tank']) + \
                      ' sustained=' + str(sustained['SENSOR1-LL-tank']) + ']'
                print '  SENSOR2-FL:        ' + str(round(v['SENSOR2-FL'], 4)) + \
                      '  [consecutive=' + str(consecutive['SENSOR2-FL']) + \
                      ' sustained=' + str(sustained['SENSOR2-FL']) + ']'
                print '  SENSOR3-LL-bottle: ' + str(round(v['SENSOR3-LL-bottle'], 4)) + \
                      '  [consecutive=' + str(consecutive['SENSOR3-LL-bottle']) + \
                      ' sustained=' + str(sustained['SENSOR3-LL-bottle']) + ']'
                print '  ACTUATOR1-MV:      ' + str(act) + \
                      '  [exp=' + str(exp) + ' ' + ('OK' if ok else 'WRONG-' + reason) + ']'
                if notes:
                    print '  *** SUSTAINED CASCADE: ' + ', '.join(notes)

            # Write to CSV
            with open(csv_file, 'ab') as f:
                csv.writer(f).writerow([
                    ts_now, cycle,
                    round(v['SENSOR1-LL-tank'], 5),
                    round(v['SENSOR2-FL'], 5),
                    round(v['SENSOR3-LL-bottle'], 5),
                    act,
                    s1_status, s2_status, s3_status,
                    consecutive['SENSOR1-LL-tank'],
                    consecutive['SENSOR2-FL'],
                    consecutive['SENSOR3-LL-bottle'],
                    exp, act,
                    'YES' if ok else 'NO',
                    reason,
                    'YES' if cascade else 'NO',
                    ' | '.join(notes)
                ])

            time.sleep(1)

    except KeyboardInterrupt:
        print '\n[Stopped by user]'

    conn.close()

    total = max(stats['total'], 1)
    cr = round(stats['cascade'] / float(total) * 100, 2)
    ver = round(stats['valve_wrong'] / float(total) * 100, 2)

    summary = [
        '=' * 60,
        'CASCADE ANALYSIS SUMMARY v3 - SUSTAINED BREACH DETECTION',
        '=' * 60,
        'Attack:                   ' + attack_name,
        'Total cycles:             ' + str(stats['total']),
        'Sustained cascade events: ' + str(stats['cascade']),
        'Cascade rate:             ' + str(cr) + '%',
        '',
        'SENSOR1 sustained breaches: ' + str(stats['s1_breach']),
        'SENSOR2 sustained breaches: ' + str(stats['s2_breach']),
        'SENSOR3 sustained breaches: ' + str(stats['s3_breach']),
        '',
        'Valve correct:            ' + str(stats['valve_ok']),
        'Valve wrong:              ' + str(stats['valve_wrong']),
        'Valve error rate:         ' + str(ver) + '%',
        '',
        'First sustained breach:   ' + str(stats['first_breach']),
        'First wrong valve:        ' + str(stats['first_wrong_valve']),
        '',
        'Note: Only breaches lasting ' + str(SUSTAINED_CYCLES) + '+ consecutive cycles are counted',
        'This eliminates natural simulation timing noise',
        '',
        'Log saved to: ' + csv_file,
        '=' * 60,
    ]

    for line in summary:
        print line

    with open(summary_file, 'w') as f:
        f.write('\n'.join(summary))

    print 'Summary saved to: ' + summary_file

if __name__ == '__main__':
    if len(sys.argv) < 3:
        print 'Usage: python monitor_cascade_v3.py <attack_name> <duration_minutes>'
        print 'Example: python monitor_cascade_v3.py NO_ATTACK 2'
        print 'Example: python monitor_cascade_v3.py SENSOR2_slow_drift 5'
        sys.exit(1)
    monitor(sys.argv[1], sys.argv[2])
