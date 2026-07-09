"""
Native SWaT-style physical process.

This replaces the small original process with a Stage-1/Stage-3
SWaT-inspired physical process.

It directly generates:
LIT101, FIT101, MV101, P101, P102,
LIT301, FIT301, MV301, P301, P302
"""

from minicps.devices import Tank

from utils import STATE
from utils import PP_PERIOD_SEC
from utils import LIT101_M, FIT101_M, LIT301_M, FIT301_M
from utils import ACT_OFF, ACT_ON

import time
import random

LIT101 = ('LIT101', 1)
FIT101 = ('FIT101', 2)
MV101 = ('MV101', 1)
P101 = ('P101', 1)
P102 = ('P102', 1)

LIT301 = ('LIT301', 3)
FIT301 = ('FIT301', 2)
MV301 = ('MV301', 3)
P301 = ('P301', 3)
P302 = ('P302', 3)


class SWaTNativeProcess(Tank):

    def pre_loop(self):
        print('DEBUG phys-proc: Native SWaT-style process booting')

        self.lit101 = 550.0
        self.lit301 = 820.0

        self.set(LIT101, self.lit101)
        self.set(FIT101, 0.0)
        self.set(MV101, ACT_OFF)
        self.set(P101, ACT_OFF)
        self.set(P102, ACT_OFF)

        self.set(LIT301, self.lit301)
        self.set(FIT301, 0.0)
        self.set(MV301, ACT_OFF)
        self.set(P301, ACT_OFF)
        self.set(P302, ACT_OFF)

    def clamp(self, value, low, high):
        return max(low, min(high, value))

    def main_loop(self):
        while True:
            mv101 = int(float(self.get(MV101)))
            p101 = int(float(self.get(P101)))

            mv301 = int(float(self.get(MV301)))
            p301 = int(float(self.get(P301)))

            # Stage 1 flow: active when valve and pump are on.
            if mv101 == ACT_ON and p101 == ACT_ON:
                fit101 = random.uniform(2.10, FIT101_M['UpperBound'])
                lit101_delta_in = random.uniform(1.0, 2.2)
            else:
                fit101 = random.uniform(0.0, 0.05)
                lit101_delta_in = 0.0

            # Stage 3 flow/draw.
            if mv301 == ACT_ON and p301 == ACT_ON:
                fit301 = random.uniform(1.80, FIT301_M['UpperBound'])
                lit101_delta_out = random.uniform(1.0, 2.0)
                lit301_delta_in = random.uniform(0.8, 1.6)
            else:
                fit301 = random.uniform(0.0, 0.05)
                lit101_delta_out = random.uniform(0.0, 0.4)
                lit301_delta_in = 0.0

            # Update levels.
            self.lit101 = (
                self.lit101
                + lit101_delta_in
                - lit101_delta_out
                + random.uniform(-0.4, 0.4)
            )

            self.lit301 = (
                self.lit301
                + lit301_delta_in
                - random.uniform(0.5, 1.2)
                + random.uniform(-0.3, 0.3)
            )

            self.lit101 = self.clamp(
                self.lit101,
                LIT101_M['LowerBound'],
                LIT101_M['UpperBound']
            )

            self.lit301 = self.clamp(
                self.lit301,
                LIT301_M['LowerBound'],
                LIT301_M['UpperBound']
            )

            self.set(LIT101, self.lit101)
            self.set(FIT101, fit101)

            self.set(LIT301, self.lit301)
            self.set(FIT301, fit301)

            print(
                'DEBUG phys-proc: '
                'LIT101 %.3f FIT101 %.3f MV101 %d P101 %d | '
                'LIT301 %.3f FIT301 %.3f MV301 %d P301 %d'
                % (
                    self.lit101,
                    fit101,
                    mv101,
                    p101,
                    self.lit301,
                    fit301,
                    mv301,
                    p301
                )
            )

            time.sleep(PP_PERIOD_SEC)

    def _stop(self):
        print('physical process stopped (SWaT native)')


if __name__ == '__main__':
    process = SWaTNativeProcess(
        name='swat_native_process',
        state=STATE,
        protocol=None,
        section=1.0,
        level=1.0
    )
