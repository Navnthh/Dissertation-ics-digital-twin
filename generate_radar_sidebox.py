#!/usr/bin/env python3
import matplotlib.pyplot as plt
import numpy as np
import os

OUTPUT_DIR = os.path.expanduser('~/cascade_graphs')
os.makedirs(OUTPUT_DIR, exist_ok=True)

data = {
    'No Attack':        {'vals': [53.0, 63.2, 49.7, 76.7],
                         'means': ['3.22m', '1.89m3/h', '0.45m', '0.77']},
    'SENSOR1 1% Drift': {'vals': [45.0, 67.3, 53.3, 82.0],
                         'means': ['2.78m', '2.02m3/h', '0.48m', '0.82']},
    'SENSOR3 1% Drift': {'vals': [48.5, 78.1, 48.7, 94.2],
                         'means': ['2.97m', '2.34m3/h', '0.44m', '0.94']},
}

categories = ['SENSOR1\nTank Level', 'SENSOR2\nFlow Rate',
              'SENSOR3\nBottle Level', 'ACTUATOR\nValve']

N = len(categories)
angles = [n / float(N) * 2 * np.pi for n in range(N)]
angles += angles[:1]

colors = {
    'No Attack':        ('#1f77b4', 'o'),
    'SENSOR1 1% Drift': ('#d62728', 's'),
    'SENSOR3 1% Drift': ('#2ca02c', '^'),
}

attack_colors_bg = {
    'No Attack':        '#d6eaf8',
    'SENSOR1 1% Drift': '#fadbd8',
    'SENSOR3 1% Drift': '#d5f5e3',
}

def draw_radar_sidebox(datasets, title, output_name):

    # wide figure — radar on left, table on right
    fig = plt.figure(figsize=(16, 8))
    fig.suptitle(title, fontsize=13, fontweight='bold', y=1.01)

    # radar chart — left side
    ax = fig.add_axes([0.03, 0.05, 0.55, 0.90], projection='polar')

    for attack, info in datasets.items():
        vals  = info['vals']
        color, marker = colors[attack]
        v = vals + vals[:1]

        ax.plot(angles, v, color=color, linewidth=2.5,
                linestyle='solid', label=attack)
        ax.fill(angles, v, color=color, alpha=0.10)
        ax.plot(angles, v, color=color, marker=marker,
                markersize=9, markerfacecolor=color,
                markeredgecolor='white', markeredgewidth=1.5,
                linestyle='none')

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=11, fontweight='bold')
    ax.tick_params(axis='x', pad=18)
    ax.set_ylim(0, 120)
    ax.set_yticks([20, 40, 60, 80, 100])
    ax.set_yticklabels(['20%', '40%', '60%', '80%', '100%'],
                       fontsize=8, color='grey')
    ax.legend(loc='upper right',
              bbox_to_anchor=(1.15, 1.15), fontsize=9)
    ax.grid(color='grey', linestyle='--', linewidth=0.5, alpha=0.5)

    # table — right side
    ax_table = fig.add_axes([0.62, 0.05, 0.36, 0.90])
    ax_table.axis('off')

    col_labels = ['Attack Type',
                  'SENSOR1\nTank\n(mean)',
                  'SENSOR2\nFlow\n(mean)',
                  'SENSOR3\nBottle\n(mean)',
                  'ACTUATOR\nValve\n(mean)']

    table_data = []
    for attack, info in datasets.items():
        row = [attack]
        for j in range(4):
            row.append(info['means'][j] +
                      '\n(' + str(info['vals'][j]) + '%)')
        table_data.append(row)

    table = ax_table.table(
        cellText=table_data,
        colLabels=col_labels,
        cellLoc='center',
        loc='center',
        bbox=[0, 0, 1, 1]
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)

    # header row
    for j in range(len(col_labels)):
        table[0, j].set_facecolor('#2c3e50')
        table[0, j].set_text_props(color='white', fontweight='bold')

    # data rows
    for i, (attack, _) in enumerate(datasets.items()):
        bg = attack_colors_bg[attack]
        for j in range(len(col_labels)):
            table[i+1, j].set_facecolor(bg)
            table[i+1, j].set_height(0.25)

    # table title
    ax_table.set_title('Mean Values at Each Point',
                      fontsize=11, fontweight='bold', pad=10)

    out = os.path.join(OUTPUT_DIR, output_name)
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print('Saved: ' + out)

# Graph 1 — SENSOR1 vs No Attack
draw_radar_sidebox(
    {'No Attack':        data['No Attack'],
     'SENSOR1 1% Drift': data['SENSOR1 1% Drift']},
    'Radar Chart — SENSOR1 1% Drift vs No Attack',
    '61_radar_sensor1_sidebox.png'
)

# Graph 2 — SENSOR3 vs No Attack
draw_radar_sidebox(
    {'No Attack':        data['No Attack'],
     'SENSOR3 1% Drift': data['SENSOR3 1% Drift']},
    'Radar Chart — SENSOR3 1% Drift vs No Attack',
    '62_radar_sensor3_sidebox.png'
)

print('Done!')
