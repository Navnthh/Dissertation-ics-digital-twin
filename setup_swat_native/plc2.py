"""
Native SWaT-style PLC2.

PLC2 publishes flow indicators:
- FIT101
- FIT301
"""

from minicps.devices import PLC

from utils import PLC2_DATA, STATE
from utils import PLC2_PROTOCOL, PLC2_ADDR
from utils import PLC_PERIOD_SEC

import time
import logging

FIT101 = ('FIT101', 2)
FIT301 = ('FIT301', 2)


class SWaTPLC2(PLC):

    def pre_loop(self, sleep=0.6):
        print 'DEBUG: SWaT PLC2 enters pre_loop'
        print
        time.sleep(sleep)

    def main_loop(self, sleep=0.0):
        print 'DEBUG: SWaT PLC2 enters main_loop.'
        print

        logging.basicConfig(
            filename='logs/swat_plc2.log',
            format='%(levelname)s %(asctime)s ' + PLC2_ADDR + ' %(funcName)s %(message)s',
            datefmt='%m/%d/%Y %H:%M:%S',
            level=logging.DEBUG
        )

        while True:
            fit101 = float(self.get(FIT101))
            fit301 = float(self.get(FIT301))

            try:
                self.send(FIT101, fit101, PLC2_ADDR)
                self.send(FIT301, fit301, PLC2_ADDR)

                print "DEBUG PLC2 - FIT101 %.5f FIT301 %.5f" % (fit101, fit301)

                logging.info("FIT101 updated: %.5f" % fit101)
                logging.info("FIT301 updated: %.5f" % fit301)
            except:
                logging.info("Could not update PLC2 ENIP flow tags")

            time.sleep(PLC_PERIOD_SEC)

        print 'DEBUG SWaT PLC2 shutdown'


if __name__ == "__main__":
    plc2 = SWaTPLC2(
        name='plc2',
        state=STATE,
        protocol=PLC2_PROTOCOL,
        memory=PLC2_DATA,
        disk=PLC2_DATA
    )
