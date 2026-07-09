import os
import sqlite3

from utils import PATH, SCHEMA, SCHEMA_INIT

if os.path.exists(PATH):
    os.remove(PATH)

conn = sqlite3.connect(PATH)
cursor = conn.cursor()

cursor.executescript(SCHEMA)
cursor.executescript(SCHEMA_INIT)

conn.commit()
conn.close()

print("Initialized SWaT-native MiniCPS database:", PATH)
