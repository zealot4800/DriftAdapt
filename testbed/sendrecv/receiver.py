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
import argparse

MAX_PKT_COUNT = 100
run_pkt_count = 0
pred_outputs = list()
true_labels = list()
f1_array  = list()


def to_float(x, e):
    c = abs(x)
    sign = 1
    if x < 0:
        # convert back from two's complement
        c = x - 1
        c = ~c
        sign = -1
    f = (1.0 * c) / (2 ** e)
    f = f * sign
    return f


# Parse received packets
def parse_output(pkt,log_file):
    if Ether in pkt:
        data = pkt[Ether].payload
        # print()
        data = bytes(data)

        if len(data) < 35:
            return

        label  = data[32]
        output = data[33]
        valid_packet = data[34]


        global run_pkt_count, pred_outputs, true_labels

        if(valid_packet == 255):
            run_pkt_count += 1

            pred_outputs.append(output)
            true_labels.append(label)

            # Calculate metrics for every MAX_PKT_COUNT packets
            if run_pkt_count == MAX_PKT_COUNT:
                print("Received {0} packets.".format(MAX_PKT_COUNT))
                calc_metrics(pred_outputs, true_labels)
                run_pkt_count = 0

                with open(log_file, 'a') as file:
                    file.write(str(f1_array[-1]) + "\n")

                # Metrics are per window, not cumulative across the full run.
                pred_outputs.clear()
                true_labels.clear()


# Computing DNN inference accuracy
def calc_metrics(pred_outputs, true_labels):
    global f1_array
    sample_count = len(true_labels)
    if sample_count == 0 or len(pred_outputs) != sample_count:
        raise ValueError('predictions and labels must be non-empty and the same length')

    accuracy = 100 * sum(
        prediction == label
        for prediction, label in zip(pred_outputs, true_labels)
    ) / sample_count
    precision_sum = 0.0
    recall_sum = 0.0
    f1_sum = 0.0
    for label in (0, 1):
        true_positive = sum(p == label and t == label for p, t in zip(pred_outputs, true_labels))
        false_positive = sum(p == label and t != label for p, t in zip(pred_outputs, true_labels))
        false_negative = sum(p != label and t == label for p, t in zip(pred_outputs, true_labels))
        precision_for_label = true_positive / (true_positive + false_positive) if true_positive + false_positive else 0.0
        recall_for_label = true_positive / (true_positive + false_negative) if true_positive + false_negative else 0.0
        f1_for_label = (
            2 * precision_for_label * recall_for_label / (precision_for_label + recall_for_label)
            if precision_for_label + recall_for_label
            else 0.0
        )
        precision_sum += precision_for_label
        recall_sum += recall_for_label
        f1_sum += f1_for_label

    precision = 100 * precision_sum / 2
    recall = 100 * recall_sum / 2
    f1 = 100 * f1_sum / 2

    print("Accuracy Across 2 Classes: {0:.2f}".format(accuracy))
    print("Macro Precision Across 2 Classes: {0:.2f}".format(precision))
    print("Macro Recall Across 2 Classes: {0:.2f}".format(recall))
    print("Macro F1-Score Across 2 Classes: {0:.2f}".format(f1))

    f1_array.append(f1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--iface', type=str, required=True)
    parser.add_argument('--logfile', type=str, required=True)
    parser.add_argument('--window-size', type=int, default=100)

    args = parser.parse_args()
    if args.window_size <= 0:
        parser.error('--window-size must be greater than zero')
    global MAX_PKT_COUNT
    MAX_PKT_COUNT = args.window_size
    iface = args.iface
    log_file = args.logfile

    with open(log_file, 'w') as file:
        file.write("F1 Scores\n")

    # sniff() runs continuously when count is omitted.
    sniff(iface=iface, prn=lambda packet: parse_output(packet, log_file), store=False)


if __name__ == "__main__":

    main()
