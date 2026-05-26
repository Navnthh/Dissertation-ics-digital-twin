#!/usr/bin/env python3
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

LOG_DIR = os.path.expanduser('~/cascade_logs')
OUTPUT_DIR = os.path.expanduser('~/cascade_graphs')
os.makedirs(OUTPUT_DIR, exist_ok=True)

def load_csv(filepath):
    try:
        df = pd.read_csv(filepath)
        df.columns = df.columns.str.strip()
        return df
    except Exception as e:
        print('Error: ' + str(e))
        return None

def to_percent(values, min_val, max_val):
    return np.array([(v - min_val) / (max_val - min_val) * 100 for v in values])

def plot_s3_1pct(attack_file, cascade_file, normal_file, output_name):
    attack_df  = load_csv(attack_file)
    cascade_df = load_csv(cascade_file)
    normal_df  = load_csv(normal_file)

    if attack_df is None or cascade_df is None or normal_df is None:
        print('Missing file')
        return

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle('SENSOR3 1% Range Drift Attack — All Sensors (%)\nNormal vs During Attack',
                fontsize=14, fontweight='bold')

    ranges = {
        'SENSOR1': (0.3,  5.81),
        'SENSOR2': (0.0,  3.0),
        'SENSOR3': (0.0,  0.9),
        'ACTUATOR':(0.0,  1.0),
    }

    # SENSOR1 — normal vs during attack
    ax1 = axes[0][0]
    min_v, max_v = ranges['SENSOR1']
    normal_pct = to_percent(normal_df['SENSOR1'].values, min_v, max_v)
    attack_pct = to_percent(cascade_df['SENSOR1'].values, min_v, max_v)
    ax1.plot(normal_df['cycle'].values, normal_pct,
             color='#4db8ff', linewidth=2,
             label='Normal %', alpha=0.9)
    ax1.plot(cascade_df['cycle'].values, attack_pct,
             color='#f85149', linewidth=2,
             label='During attack %', linestyle='--', alpha=0.9)
    ax1.axhline(y=100, color='red', linewidth=1,
               linestyle=':', alpha=0.7, label='Upper 100%')
    ax1.axhline(y=0, color='orange', linewidth=1,
               linestyle=':', alpha=0.7, label='Lower 0%')
    ax1.set_ylim(-10, 110)
    ax1.set_title('SENSOR1 — Tank Level (%)',
                 fontsize=11, fontweight='bold')
    ax1.set_xlabel('Cycle (seconds)', fontsize=9)
    ax1.set_ylabel('% of range', fontsize=9)
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)
    ax1.set_facecolor('#f8f9fa')

    # SENSOR2 — normal vs during attack
    ax2 = axes[0][1]
    min_v, max_v = ranges['SENSOR2']
    normal_pct = to_percent(normal_df['SENSOR2'].values, min_v, max_v)
    attack_pct = to_percent(cascade_df['SENSOR2'].values, min_v, max_v)
    ax2.plot(normal_df['cycle'].values, normal_pct,
             color='#4dff9a', linewidth=2,
             label='Normal %', alpha=0.9)
    ax2.plot(cascade_df['cycle'].values, attack_pct,
             color='#f85149', linewidth=2,
             label='During attack %', linestyle='--', alpha=0.9)
    ax2.axhline(y=100, color='red', linewidth=1,
               linestyle=':', alpha=0.7, label='Upper 100%')
    ax2.axhline(y=0, color='orange', linewidth=1,
               linestyle=':', alpha=0.7, label='Lower 0%')
    ax2.set_ylim(-10, 110)
    ax2.set_title('SENSOR2 — Flow Rate (%)',
                 fontsize=11, fontweight='bold')
    ax2.set_xlabel('Cycle (seconds)', fontsize=9)
    ax2.set_ylabel('% of range', fontsize=9)
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)
    ax2.set_facecolor('#f8f9fa')

    # SENSOR3 — real vs fake in percentage
    ax3 = axes[1][0]
    min_v, max_v = ranges['SENSOR3']
    real_pct = to_percent(attack_df['real_value'].values, min_v, max_v)
    fake_pct = to_percent(attack_df['fake_value'].values, min_v, max_v)
    ax3.plot(attack_df['cycle'].values, real_pct,
             color='#ffd94d', linewidth=2,
             label='Real value %', alpha=0.9)
    ax3.plot(attack_df['cycle'].values, fake_pct,
             color='#f85149', linewidth=2,
             label='Fake value % (1% drift)',
             linestyle='--', alpha=0.9)
    ax3.axhline(y=100, color='red', linewidth=1,
               linestyle=':', alpha=0.7, label='Upper 100%')
    ax3.axhline(y=0, color='orange', linewidth=1,
               linestyle=':', alpha=0.7, label='Lower 0%')
    ax3.set_ylim(-10, 110)
    ax3.set_title('SENSOR3 — Bottle Level (%) — ATTACKED',
                 fontsize=11, fontweight='bold')
    ax3.set_xlabel('Cycle (seconds)', fontsize=9)
    ax3.set_ylabel('% of range', fontsize=9)
    ax3.legend(fontsize=8)
    ax3.grid(True, alpha=0.3)
    ax3.set_facecolor('#fff0f0')

    # ACTUATOR — normal vs during attack
    ax4 = axes[1][1]
    min_v, max_v = ranges['ACTUATOR']
    normal_pct = to_percent(normal_df['ACTUATOR'].values, min_v, max_v)
    attack_pct = to_percent(cascade_df['ACTUATOR'].values, min_v, max_v)
    ax4.plot(normal_df['cycle'].values, normal_pct,
             color='#c080ff', linewidth=2,
             label='Normal %', alpha=0.9)
    ax4.plot(cascade_df['cycle'].values, attack_pct,
             color='#f85149', linewidth=2,
             label='During attack %', linestyle='--', alpha=0.9)
    ax4.axhline(y=100, color='red', linewidth=1,
               linestyle=':', alpha=0.7, label='Upper 100%')
    ax4.axhline(y=0, color='orange', linewidth=1,
               linestyle=':', alpha=0.7, label='Lower 0%')
    ax4.set_ylim(-10, 110)
    ax4.set_title('ACTUATOR — Valve (%)',
                 fontsize=11, fontweight='bold')
    ax4.set_xlabel('Cycle (seconds)', fontsize=9)
    ax4.set_ylabel('% of range', fontsize=9)
    ax4.legend(fontsize=8)
    ax4.grid(True, alpha=0.3)
    ax4.set_facecolor('#f8f9fa')

    plt.tight_layout()
    out = os.path.join(OUTPUT_DIR, output_name + '.png')
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print('Saved: ' + out)

plot_s3_1pct(
    os.path.join(LOG_DIR, 'slow_refill_1pct_SENSOR3_20260519_025940.csv'),
    os.path.join(LOG_DIR, 'cascade_v3_SENSOR3_1pct_anystart_20260519_025953.csv'),
    os.path.join(LOG_DIR, 'cascade_v3_NO_ATTACK_5min_20260507_040228.csv'),
    '46_SENSOR3_1pct_anystart_percent'
)
print('Done!')
