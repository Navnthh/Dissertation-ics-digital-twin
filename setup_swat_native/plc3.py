"""
Native SWaT-style PLC3.

PLC3 handles downstream Stage 3:
- reads LIT301
- controls MV301, P301, P302
"""

from minicps.devices import PLC

from utils import PLC3_DATA, STATE
from utils import PLC3_PROTOCOL, PLC3_ADDR
from utils import PLC_PERIOD_SEC
from utils import LIT301_M, ACT_OFF, ACT_ON

import time
import logging

LIT301 = ('LIT301', 3)
FIT301 = ('FIT301', 2)

MV301 = ('MV301', 3)
P301 = ('P301', 3)
P302 = ('P302', 3)


class SWaTPLC3(PLC):

    def pre_loop(self, sleep=0.6):
        print 'DEBUG: SWaT PLC3 enters pre_loop'
        print
        time.sleep(sleep)

    def main_loop(self, sleep=0.0):
        print 'DEBUG: SWaT PLC3 enters main_loop.'
        print

        logging.basicConfig(
            filename='logs/swat_plc3.log',
            format='%(levelname)s %(asctime)s ' + PLC3_ADDR + ' %(funcName)s %(message)s',
            datefmt='%m/%d/%Y %H:%M:%S',
            level=logging.DEBUG
        )

        while True:
            lit301 = float(self.get(LIT301))

            if lit301 <= LIT301_M['LowControl']:
                mv301 = ACT_ON
                p301 = ACT_ON
            elif lit301 >= LIT301_M['HighControl']:
                mv301 = ACT_OFF
                p301 = ACT_OFF
            else:
                mv301 = ACT_ON
                p301 = ACT_ON

            p302 = ACT_OFF

            self.set(MV301, mv301)
            self.set(P301, p301)
            self.set(P302, p302)

            try:
                self.send(LIT301, lit301, PLC3_ADDR)
                self.send(MV301, mv301, PLC3_ADDR)
                self.send(P301, p301, PLC3_ADDR)
                self.send(P302, p302, PLC3_ADDR)

                print "DEBUG PLC3 - LIT301 %.5f MV301 %d P301 %d" % (lit301, mv301, p301)

                logging.info("LIT301 updated: %.5f" % lit301)
            except:
                logging.info("Could not update PLC3 ENIP tags")

            time.sleep(PLC_PERIOD_SEC)

        print 'DEBUG SWaT PLC3 shutdown'


if __name__ == "__main__":
    plc3 = SWaTPLC3(
        name='plc3',
        state=STATE,
        protocol=PLC3_PROTOCOL,
        memory=PLC3_DATA,
        disk=PLC3_DATA
    )
