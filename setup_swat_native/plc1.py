"""
Native SWaT-style PLC1.

PLC1 controls Stage 1:
- reads LIT101
- receives FIT101 from PLC2
- receives LIT301/FIT301 from PLC3
- controls MV101, P101, P102
"""

from minicps.devices import PLC

from utils import PLC1_DATA, PLC1_PROTOCOL, PLC1_ADDR, STATE
from utils import PLC2_ADDR, PLC3_ADDR
from utils import PLC_PERIOD_SEC
from utils import LIT101_M, FIT101_M, LIT301_M
from utils import ACT_OFF, ACT_ON

import time
import logging
import csv
import datetime
import os.path

LIT101 = ('LIT101', 1)
FIT101_1 = ('FIT101', 1)
FIT101_2 = ('FIT101', 2)

MV101 = ('MV101', 1)
P101 = ('P101', 1)
P102 = ('P102', 1)

LIT301_1 = ('LIT301', 1)
LIT301_3 = ('LIT301', 3)

FIT301_1 = ('FIT301', 1)
FIT301_2 = ('FIT301', 2)

MV301_1 = ('MV301', 1)
P301_1 = ('P301', 1)


class SWaTPLC1(PLC):

    def pre_loop(self, sleep=0.1):
        print('DEBUG: SWaT PLC1 enters pre_loop')
        print()
        time.sleep(sleep)

    def store_values(self, lit101, fit101, lit301, fit301, mv101, p101, count):
        with open('logs/swat_native_data.csv', 'a+') as writeobj:
            fieldnames = [
                'timestamp',
                'LIT101',
                'FIT101',
                'MV101',
                'P101',
                'LIT301',
                'FIT301'
            ]

            csv_writer = csv.DictWriter(writeobj, fieldnames=fieldnames)

            if count == 0:
                csv_writer.writeheader()

            csv_writer.writerow({
                'timestamp': str(datetime.datetime.now()),
                'LIT101': lit101,
                'FIT101': fit101,
                'MV101': mv101,
                'P101': p101,
                'LIT301': lit301,
                'FIT301': fit301
            })

    def main_loop(self):
        print('DEBUG: SWaT PLC1 enters main_loop.')
        print()

        logging.basicConfig(
            filename='logs/swat_plc1.log',
            format='%(levelname)s %(asctime)s ' + PLC1_ADDR + ' %(funcName)s %(message)s',
            datefmt='%m/%d/%Y %H:%M:%S',
            level=logging.DEBUG
        )

        count = 0

        while True:
            lit101 = float(self.get(LIT101))
            print('DEBUG PLC1 - LIT101: %.5f' % lit101)
            self.send(LIT101, lit101, PLC1_ADDR)

            try:
                fit101 = float(self.receive(FIT101_2, PLC2_ADDR))
                print('DEBUG PLC1 - received FIT101: %.5f' % fit101)
                self.send(FIT101_1, fit101, PLC1_ADDR)
            except:
                logging.warning("FIT101 not received from PLC2")
                fit101 = 999.0

            try:
                lit301 = float(self.receive(LIT301_3, PLC3_ADDR))
                fit301 = float(self.receive(FIT301_2, PLC2_ADDR))

                print('DEBUG PLC1 - received LIT301: %.5f FIT301: %.5f' % (lit301, fit301))

                self.send(LIT301_1, lit301, PLC1_ADDR)
                self.send(FIT301_1, fit301, PLC1_ADDR)
            except:
                logging.warning("Stage 3 values not received")
                lit301 = 999.0
                fit301 = 999.0

            # Stage 1 control logic.
            if lit101 <= LIT101_M['LowControl']:
                print('INFO PLC1 - LIT101 low -> open MV101 and start P101')
                self.set(MV101, ACT_ON)
                self.set(P101, ACT_ON)

                self.send(MV101, ACT_ON, PLC1_ADDR)
                self.send(P101, ACT_ON, PLC1_ADDR)

            elif lit101 >= LIT101_M['HighControl']:
                print('INFO PLC1 - LIT101 high -> close MV101 and stop P101')
                self.set(MV101, ACT_OFF)
                self.set(P101, ACT_OFF)

                self.send(MV101, ACT_OFF, PLC1_ADDR)
                self.send(P101, ACT_OFF, PLC1_ADDR)

            elif lit301 >= LIT301_M['HighControl']:
                print('INFO PLC1 - LIT301 high -> close MV101 and stop P101')
                self.set(MV101, ACT_OFF)
                self.set(P101, ACT_OFF)

                self.send(MV101, ACT_OFF, PLC1_ADDR)
                self.send(P101, ACT_OFF, PLC1_ADDR)

            else:
                print('INFO PLC1 - normal band -> keep MV101 and P101 active')
                self.set(MV101, ACT_ON)
                self.set(P101, ACT_ON)

                self.send(MV101, ACT_ON, PLC1_ADDR)
                self.send(P101, ACT_ON, PLC1_ADDR)

            self.set(P102, ACT_OFF)
            self.send(P102, ACT_OFF, PLC1_ADDR)

            mv101 = int(float(self.get(MV101)))
            p101 = int(float(self.get(P101)))

            if os.path.isfile('trigger.txt'):
                self.store_values(lit101, fit101, lit301, fit301, mv101, p101, count)
                count = 1

            time.sleep(PLC_PERIOD_SEC)

        print('DEBUG SWaT PLC1 shutdown')


if __name__ == "__main__":
    plc1 = SWaTPLC1(
        name='plc1',
        state=STATE,
        protocol=PLC1_PROTOCOL,
        memory=PLC1_DATA,
        disk=PLC1_DATA
    )
