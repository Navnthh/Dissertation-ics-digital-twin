"""
Native SWaT-style MiniCPS utility configuration.

This setup expands the original MiniCPS filling-plant model into
SWaT-inspired Stage 1 and Stage 3 components.

Core SWaT-style tags:
- LIT101: Stage 1 tank level
- FIT101: Stage 1 flow
- MV101: Stage 1 motorized valve
- P101/P102: Stage 1 pumps
- LIT301: Stage 3 downstream level
- FIT301: Stage 3 flow
- MV301: Stage 3 motorized valve
- P301/P302: Stage 3 pumps
"""

from minicps.utils import build_debug_logger

fp_logger = build_debug_logger(
    name=__name__,
    bytes_per_file=10000,
    rotating_files=2,
    lformat='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    ldir='logs/',
    suffix='')

# ---------------------------------------------------------------------
# Initial ENIP data
# ---------------------------------------------------------------------

PLC1_DATA = {
    'LIT101': '550.0',
    'FIT101': '0.0',
    'MV101': '1',
    'P101': '1',
    'P102': '1',
    'LIT301': '820.0',
    'FIT301': '0.0',
    'MV301': '1',
    'P301': '1',
    'P302': '1'
}

PLC2_DATA = {
    'FIT101': '0.0',
    'FIT301': '0.0'
}

PLC3_DATA = {
    'LIT301': '820.0',
    'FIT301': '0.0',
    'MV301': '1',
    'P301': '1',
    'P302': '1'
}

HMI_DATA = {
    'MV101': '1',
    'P101': '1',
    'MV301': '1',
    'P301': '1'
}

# ---------------------------------------------------------------------
# PLC network configuration
# ---------------------------------------------------------------------

PLC1_MAC = '00:00:00:00:00:01'
PLC1_ADDR = '10.0.0.1'

PLC1_TAGS = (
    ('LIT101', 1, 'REAL'),
    ('FIT101', 1, 'REAL'),
    ('MV101', 1, 'INT'),
    ('P101', 1, 'INT'),
    ('P102', 1, 'INT'),
    ('LIT301', 1, 'REAL'),
    ('FIT301', 1, 'REAL'),
    ('MV301', 1, 'INT'),
    ('P301', 1, 'INT'),
    ('P302', 1, 'INT'),
)

PLC1_SERVER = {
    'address': PLC1_ADDR,
    'tags': PLC1_TAGS
}

PLC1_PROTOCOL = {
    'name': 'enip',
    'mode': 1,
    'server': PLC1_SERVER
}

PLC2_MAC = '00:00:00:00:00:02'
PLC2_ADDR = '10.0.0.2'

PLC2_TAGS = (
    ('FIT101', 2, 'REAL'),
    ('FIT301', 2, 'REAL'),
)

PLC2_SERVER = {
    'address': PLC2_ADDR,
    'tags': PLC2_TAGS
}

PLC2_PROTOCOL = {
    'name': 'enip',
    'mode': 1,
    'server': PLC2_SERVER
}

PLC3_MAC = '00:00:00:00:00:03'
PLC3_ADDR = '10.0.0.3'

PLC3_TAGS = (
    ('LIT301', 3, 'REAL'),
    ('FIT301', 3, 'REAL'),
    ('MV301', 3, 'INT'),
    ('P301', 3, 'INT'),
    ('P302', 3, 'INT'),
)

PLC3_SERVER = {
    'address': PLC3_ADDR,
    'tags': PLC3_TAGS
}

PLC3_PROTOCOL = {
    'name': 'enip',
    'mode': 1,
    'server': PLC3_SERVER
}

HMI_MAC = '00:00:00:00:00:04'
HMI_ADDR = '10.0.0.4'

HMI_TAGS = (
    ('MV101', 4, 'INT'),
    ('P101', 4, 'INT'),
    ('MV301', 4, 'INT'),
    ('P301', 4, 'INT'),
)

HMI_SERVER = {
    'address': HMI_ADDR,
    'tags': HMI_TAGS
}

HMI_PROTOCOL = {
    'name': 'enip',
    'mode': 0,
    'server': HMI_SERVER
}

ATTCKR_MAC = '00:00:00:00:00:05'
ATTCKR_ADDR = '10.0.0.5'

NETMASK = '/24'

# ---------------------------------------------------------------------
# Timing
# ---------------------------------------------------------------------

PLC_PERIOD_SEC = 0.50
PLC_PERIOD_HOURS = PLC_PERIOD_SEC / 3600.0

PP_RESCALING_HOURS = 100
PP_PERIOD_SEC = 0.25
PP_PERIOD_HOURS = (PP_PERIOD_SEC / 3600.0) * PP_RESCALING_HOURS

HMI_PERIOD_SEC = 1

# ---------------------------------------------------------------------
# SWaT-inspired ranges from profiling
# ---------------------------------------------------------------------

LIT101_M = {
    'LowerBound': 302.678,
    'UpperBound': 814.3377,
    'LowControl': 520.0,
    'HighControl': 760.0
}

FIT101_M = {
    'LowerBound': 0.0,
    'UpperBound': 2.669827
}

LIT301_M = {
    'LowerBound': 613.7849,
    'UpperBound': 1012.521,
    'LowControl': 760.0,
    'HighControl': 960.0
}

FIT301_M = {
    'LowerBound': 0.0,
    'UpperBound': 2.251033
}

# SWaT-style actuator state convention used here:
# 1 = closed/off
# 2 = open/on
ACT_OFF = 1
ACT_ON = 2

# ---------------------------------------------------------------------
# State database
# ---------------------------------------------------------------------

PATH = 'fp_swat_native_db.sqlite'
NAME = 'fp_swat_native_table'

STATE = {
    'name': NAME,
    'path': PATH
}

SCHEMA = """
CREATE TABLE fp_swat_native_table (
    name              TEXT NOT NULL,
    pid               INTEGER NOT NULL,
    value             TEXT,
    PRIMARY KEY (name, pid)
);
"""

SCHEMA_INIT = """
    INSERT INTO fp_swat_native_table VALUES ('LIT101', 1, '550.0');
    INSERT INTO fp_swat_native_table VALUES ('FIT101', 2, '0.0');
    INSERT INTO fp_swat_native_table VALUES ('MV101', 1, '1');
    INSERT INTO fp_swat_native_table VALUES ('P101', 1, '1');
    INSERT INTO fp_swat_native_table VALUES ('P102', 1, '1');

    INSERT INTO fp_swat_native_table VALUES ('LIT301', 3, '820.0');
    INSERT INTO fp_swat_native_table VALUES ('FIT301', 2, '0.0');
    INSERT INTO fp_swat_native_table VALUES ('MV301', 3, '1');
    INSERT INTO fp_swat_native_table VALUES ('P301', 3, '1');
    INSERT INTO fp_swat_native_table VALUES ('P302', 3, '1');
"""
