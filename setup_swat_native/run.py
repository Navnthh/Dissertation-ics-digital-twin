"""
Native SWaT-style MiniCPS run.py
"""

from mininet.net import Mininet
from mininet.cli import CLI
from mininet.term import makeTerm
from minicps.mcps import MiniCPS
from topo import FPTopo

import time


class SWaTNativeCPS(MiniCPS):

    def __init__(self, name, net):
        self.name = name
        self.net = net

        net.start()
        net.pingAll()

        plc1, plc2, plc3, s1, attacker, hmi = self.net.get(
            'plc1',
            'plc2',
            'plc3',
            's1',
            'attacker',
            'hmi'
        )

        net.terms += makeTerm(s1, display=None, cmd='python physical_process.py')
        time.sleep(0.2)

        net.terms += makeTerm(plc3, display=None, cmd='python plc3.py')
        time.sleep(0.2)

        net.terms += makeTerm(plc2, display=None, cmd='python plc2.py')
        time.sleep(0.2)

        net.terms += makeTerm(plc1, display=None, cmd='python plc1.py')
        time.sleep(0.2)

        net.terms += makeTerm(plc1, display=None)
        time.sleep(0.2)

        net.terms += makeTerm(hmi, display=None)
        time.sleep(0.2)

        net.terms += makeTerm(attacker, display=None)

        CLI(self.net)


if __name__ == "__main__":
    topo = FPTopo()
    net = Mininet(topo=topo)

    cps = SWaTNativeCPS(
        name='SWaTNativeCPS',
        net=net
    )
