# *************************************************************************
#
# Copyright 2024 Qizheng Zhang (Stanford University),
#                Ali Imran (Purdue University),
#                Enkeleda Bardhi (Sapienza University of Rome),
#                Tushar Swamy (Unaffiliated),
#                Nathan Zhang (Stanford University),
#                Muhammad Shahbaz (Purdue University),
#                Kunle Olukotun (Stanford University)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# *************************************************************************

from scapy.all import *
import csv
import numpy
import argparse

# The feature header definition for storing DNN inputs
class FeatureHeader(Packet):
    name = "Feature Header"
    fields_desc=[BitField("input1", 0,16),
                 BitField("input2", 0,16),
                 BitField("input3", 0,16),
                 BitField("input4", 0,16),
                 BitField("input5", 0,16),
                 BitField("input6", 0,16),
                 BitField("input7", 0,16),
                 BitField("input8", 0,16),
                 BitField("input9", 0,16),
                 BitField("input10", 0,16),
                 BitField("input11", 0,16),
                 BitField("input12", 0,16),
                 BitField("input13", 0,16),
                 BitField("input14", 0,16),
                 BitField("input15", 0,16),
                 BitField("input16", 0,16),
                 BitField("label"  , 0,8),
                 BitField("output" , 0,8),
                 BitField("valid" , 0,8),
                ]


bind_layers(IP, FeatureHeader, proto=255)


def c_to_fix(float_value):
    # Scaling by 2^8 (256) for 8 bits fractional part
    scaled_value = float_value * 256

    # Converting to integer (rounding or truncating as necessary)
    integer_value = int(round(scaled_value))

    if not -32768 <= integer_value <= 32767:
        raise ValueError(f'{float_value} is outside the signed Q8.8 range')

    # Scapy BitField expects the unsigned representation of the 16 raw bits.
    return integer_value & 0xffff


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--iface', type=str, required=True)
    parser.add_argument('--datafile', type=str, required=True)
    parser.add_argument('--count', type=int, default=None,
                        help='Number of rows to send (default: all rows)')
    parser.add_argument('--dst-mac', default='ff:ff:ff:ff:ff:ff')

    args = parser.parse_args()
    if args.count is not None and args.count < 0:
        parser.error('--count must be zero or greater')
    iface = args.iface
    datafile=args.datafile

    with open(datafile, newline='') as stream:
        data = list(csv.reader(stream))
    result = numpy.array(data).astype("float")

    if result.ndim != 2 or result.shape[1] != 17:
        parser.error('datafile must contain exactly 16 features and one label')

    packet_count = len(result) if args.count is None else min(args.count, len(result))

    # Creates packets by using data from the CSV file
    for y in range(packet_count):
        try:
            z = [c_to_fix(x) for x in result[y][:16]]
        except ValueError as error:
            parser.error(f'row {y + 1}: {error}')
        label = result[y][16]
        if label not in (0.0, 1.0):
            parser.error(f'row {y + 1}: label must be 0 or 1, got {label}')
        k = int(label)
        pkt = Ether(dst=args.dst_mac)/FeatureHeader(input1=z[0], input2=z[1], input3=z[2], input4=z[3], input5=z[4], input6=z[5], input7=z[6], input8=z[7], input9=z[8], input10=z[9], input11=z[10], input12=z[11], input13=z[12], input14=z[13], input15=z[14], input16=z[15], label=k, output=0, valid=255)
        sendp(pkt, iface=iface, verbose=False)


if __name__ == "__main__":

    main()
