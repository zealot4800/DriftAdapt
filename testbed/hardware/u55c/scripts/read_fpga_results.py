#!/usr/bin/env python3
"""Read ABI 5 FPGA status; task metrics come from the validated host run report."""
import argparse
import json
from driftadapt_runtime import ASSETS, load_assets
from run_online_adaptation import Jtag
if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--assets',default=ASSETS)
    parser.add_argument('--vivado',default='/home/palhad/Xilinx/Vivado/2022.2/bin/vivado')
    parser.add_argument('--server',default='TCP:localhost:3121')
    args=parser.parse_args()
    print(json.dumps(Jtag(args.vivado,args.server,load_assets(args.assets)['build_id']).command('status'),indent=2))
