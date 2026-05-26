#!/usr/bin/env python3
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import os

OUTPUT_DIR = os.path.expanduser('~/cascade_graphs')
os.makedirs(OUTPUT_DIR, exist_ok=True)

fig, ax = plt.subplots(figsize=(18, 10))
ax.axis('off')
fig.suptitle('Cascading Impact Matrix — ICS Filling Plant\nAttack Source → Affected Components → Physical Problems',
            fontsize=14, fontweight='bold')

# Column headers
col_headers = [
    'Attack\nSource',
    'SENSOR1\nTank Level',
    'SENSOR2\nFlow Rate',
    'SENSOR3\nBottle Level',
    'ACTUATOR\nValve',
    'Physical\nProblem',
    'Severity'
]

# Row data
rows = [
    # No attack baseline
    [
        'No Attack\n(Baseline)',
        'Normal\n53% avg\n3.22m',
        'Normal\n63% avg\n1.89 m3/h',
        'Normal\n50% avg\n0.45m',
        'Open 77%\nof time',
        'Plant operating\nnormally',
        'NONE'
    ],
    # SENSOR1 1% drift
    [
        'SENSOR1\n1% Drift\nAttack\n(PLC1)',
        'CORRUPTED\n45% avg\n2.78m\n(-8% from normal)',
        'Slightly affected\n67% avg\n2.02 m3/h\n(+4%)',
        'Slightly affected\n53% avg\n0.48m\n(+3%)',
        'More open\n82% of time\n(+5%)',
        'Tank level\nreported lower\nPump protection\ndisabled\nPump runs dry',
        'HIGH'
    ],
    # SENSOR3 1% drift
    [
        'SENSOR3\n1% Drift\nAttack\n(PLC3)',
        'Slightly affected\n49% avg\n2.97m\n(-4%)',
        'Most affected\n78% avg\n2.34 m3/h\n(+15%)',
        'CORRUPTED\n49% avg\n0.44m\n(-1%)',
        'Most affected\nOpen 94%\nof time\n(+17%)',
        'Bottle level\nreported wrong\nValve open\nalmost always\nBottles overflow\nor underfilled',
        'HIGH'
    ],
]

# Color scheme
header_color  = '#2c3e50'
no_attack_bg  = '#eafaf1'
sensor1_bg    = '#fadbd8'
sensor3_bg    = '#d5f5e3'
corrupted_bg  = '#e74c3c'
affected_bg   = '#f9e79f'
normal_bg     = '#d5f5e3'
problem_bg    = '#fdebd0'
severity_colors = {
    'NONE': '#2ecc71',
    'HIGH': '#e74c3c',
}

row_colors = [no_attack_bg, sensor1_bg, sensor3_bg]

# cell colors per row
cell_colors = [
    # No attack
    [no_attack_bg, normal_bg,    normal_bg,    normal_bg,    normal_bg,    '#eafaf1', '#2ecc71'],
    # SENSOR1 attack
    [sensor1_bg,   corrupted_bg, affected_bg,  affected_bg,  affected_bg,  problem_bg, '#e74c3c'],
    # SENSOR3 attack
    [sensor3_bg,   affected_bg,  affected_bg,  corrupted_bg, affected_bg,  problem_bg, '#e74c3c'],
]

# text colors
text_colors = [
    ['black', 'black', 'black', 'black', 'black', 'black', 'white'],
    ['black', 'white', 'black', 'black', 'black', 'black', 'white'],
    ['black', 'black', 'black', 'white', 'black', 'black', 'white'],
]

table = ax.table(
    cellText=rows,
    colLabels=col_headers,
    cellLoc='center',
    loc='center',
    bbox=[0, 0, 1, 1]
)

table.auto_set_font_size(False)
table.set_fontsize(8.5)

# Style header
for j in range(len(col_headers)):
    table[0, j].set_facecolor(header_color)
    table[0, j].set_text_props(color='white', fontweight='bold',
                               fontsize=10)
    table[0, j].set_height(0.12)

# Style data rows
for i in range(len(rows)):
    for j in range(len(col_headers)):
        table[i+1, j].set_facecolor(cell_colors[i][j])
        table[i+1, j].set_text_props(color=text_colors[i][j],
                                     fontweight='bold')
        table[i+1, j].set_height(0.26)

# Legend
legend_elements = [
    mpatches.Patch(facecolor=corrupted_bg, edgecolor='black',
                  label='Directly Attacked/Corrupted'),
    mpatches.Patch(facecolor=affected_bg, edgecolor='black',
                  label='Cascade Affected'),
    mpatches.Patch(facecolor=normal_bg, edgecolor='black',
                  label='Normal / Unaffected'),
    mpatches.Patch(facecolor='#2ecc71', edgecolor='black',
                  label='No Problem'),
    mpatches.Patch(facecolor='#e74c3c', edgecolor='black',
                  label='High Severity'),
]
ax.legend(handles=legend_elements,
          loc='lower center',
          bbox_to_anchor=(0.5, -0.08),
          ncol=5,
          fontsize=9,
          frameon=True)

plt.tight_layout()
out = os.path.join(OUTPUT_DIR, '64_cascade_problem_matrix.png')
plt.savefig(out, dpi=150, bbox_inches='tight')
plt.close()
print('Saved: ' + out)
print('Done!')
