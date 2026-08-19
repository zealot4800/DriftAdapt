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

from influxdb import InfluxDBClient
import time
import pypci
import struct
import numpy as np
import copy
import os
from pathlib import Path
import mmap

# modules and libraries necessary for training
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import f1_score


# Converts floating point number to a fixed point number
def c_to_fix(float_value):
    # Scaling by 2^8 (256) for 8 bits fractional part
    scaled_value = float_value * 65536

    # Converting to integer (rounding or truncating as necessary)
    integer_value = int(round(scaled_value))

    # Converting to 16-bit binary representation
    # Format the integer value as a binary string with 16 bits, padding with zeros if necessary
    binary_representation = format(integer_value, '032b')
    signed_integer = int(binary_representation,2)
    binary_data = struct.pack('>i', signed_integer)  # '>h' stands for big-endian 16-bit signed integer

    # Convert binary data to hexadecimal string
    hexadecimal_string = bin(int.from_bytes(binary_data, 'big'))[:]

    # Ensure the hexadecimal string is 16 bits long by zero-padding if necessary
    # hexadecimal_string = hexadecimal_string.zfill(16)

    # print(hex(int(hexadecimal_string,2)))
    return int(hexadecimal_string,2)


# the MLP model used as the labeler of the CIC-IDS2017/2018 dataset
# input features are consistent with the pForest paper
class ID_CIC_IDS2017_large_pforest(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(16, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, 4),
            nn.ReLU(),
            nn.Linear(4, 2),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        sigmoid_score = self.linear_relu_stack(x)
        return sigmoid_score

    def run_inference(self, input_features):
        score = self.forward(input_features)
        label = torch.argmax(score, dim=1)
        return score, label


# the MLP model used in the data plane of the CIC-IDS2017/2018 dataset
# input features are consistent with the pForest paper
class ID_CIC_IDS2017_small_pforest(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, 4),
            nn.ReLU(),
            nn.Linear(4, 2),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        sigmoid_score = self.linear_relu_stack(x)
        return sigmoid_score

    def result_after_first_layer(self, x):
        return self.linear_relu_stack[1](self.linear_relu_stack[0](x))

    def result_after_second_layer(self, x):
        y = self.result_after_first_layer(x)
        return self.linear_relu_stack[3](self.linear_relu_stack[2](y))

    def result_after_third_layer(self, x):
        z = self.result_after_second_layer(x)
        return self.linear_relu_stack[4](z)

    def run_inference(self, input_features):
        score = self.forward(input_features)
        label = torch.argmax(score, dim=1)
        return score, label


# use this to generate the dataloader for online training
class OnlineDataset(Dataset):
    def __init__(self, features, labels, standardize=False, normalize=False, device='cpu'):
        self.features = features.type(torch.float32)
        self.labels = labels.type(torch.int64)

    def __getitem__(self, idx):
        return self.features[idx, :], self.labels[idx]

    def __len__(self):
        return self.features.shape[0]


# Match stimulation/src/trigger.py: retrain when the current labeler-based
# accuracy proxy falls below the configured quality threshold.
def activate_retraining_trigger(model_f1_proxy, threshold):
    return model_f1_proxy < threshold


def form_balanced_binary_training_set(features, labels):
    """Apply the stimulation pipeline's random binary undersampling."""
    classes = torch.unique(labels)
    if classes.numel() != 2 or not all(int(value) in (0, 1) for value in classes):
        return None, None
    positive = torch.nonzero(labels == 1, as_tuple=False).squeeze(1)
    negative = torch.nonzero(labels == 0, as_tuple=False).squeeze(1)
    count = min(positive.numel(), negative.numel())
    if count == 0:
        return None, None
    positive = positive[torch.randperm(positive.numel())[:count]]
    negative = negative[torch.randperm(negative.numel())[:count]]
    indices = torch.cat((negative, positive))
    return features[indices], labels[indices]


# retrain a model based on given labeled dataset
def retrain_model(my_dnn, training_dataloader, total_epochs, optimizer, loss_fn):

    my_dnn.train()

    for epoch_index in range(total_epochs):

        # process each mini-batch
        for _, (features, gt_label) in enumerate(training_dataloader):

            # clear the gradients
            optimizer.zero_grad()

            # run inference
            my_score, _ = my_dnn.run_inference(features)
            loss = loss_fn(my_score, gt_label)

            # backprop
            loss.backward()
            optimizer.step()

    return my_dnn, optimizer


# Setup access to the FPGA via PCIe
U250_VENDOR = 0x10ee
U250_GOLDEN_DEVICE = 0xd004
U250_DEVICE = 0x903f

SYS_CONFIG_BASE = 0x00000
SYS_CONFIG_HIGH = 0x01000
QDMA_SUBSYS_BASE = 0x01000
QDMA_SUBSYS_HIGH = 0x08000
CMAC_SUBSYS_BASE = 0x08000
CMAC_SUBSYS_HIGH = 0x10000
SPATIAL_SUBSYS_BASE = 0x10000
SPATIAL_SUBSYS_HIGH = 0x100000

u250_bar1 = None
u55c_resource_file = None
u55c_bar2 = None
HARDWARE_TARGET = os.getenv("DRIFTADAPT_HARDWARE_TARGET", "u55c").lower()
U55C_FEATURE_ADDRESS = 0x100000
U55C_FEATURE_WORD = 0x44524654
U55C_STATUS_ADDRESS = 0x100004


def init(vendor, device):
    global u250_bar1, u55c_resource_file, u55c_bar2
    if HARDWARE_TARGET == "u55c":
        resource = Path(os.getenv(
            "DRIFTADAPT_PCI_RESOURCE",
            "/sys/bus/pci/devices/0000:01:00.0/resource2",
        ))
        if not resource.is_file():
            raise RuntimeError(f"U55C OpenNIC BAR2 resource is missing: {resource}")
        u55c_resource_file = resource.open("r+b", buffering=0)
        size = os.fstat(u55c_resource_file.fileno()).st_size
        if size <= U55C_FEATURE_ADDRESS + 4:
            raise RuntimeError(f"U55C BAR2 is too small ({size} bytes): {resource}")
        u55c_bar2 = mmap.mmap(
            u55c_resource_file.fileno(), size, flags=mmap.MAP_SHARED,
            prot=mmap.PROT_READ | mmap.PROT_WRITE,
        )
        feature = read(U55C_FEATURE_ADDRESS)
        if feature != U55C_FEATURE_WORD:
            raise RuntimeError(
                f"DRIFTADAPT U55C feature word not found at BAR2 {U55C_FEATURE_ADDRESS:#x}: "
                f"read {feature:#010x}, expected {U55C_FEATURE_WORD:#010x}. "
                "Program the validated DRIFTADAPT U55C OpenNIC image first."
            )
        return 0
    if HARDWARE_TARGET != "u250":
        raise ValueError("DRIFTADAPT_HARDWARE_TARGET must be 'u55c' or 'u250'")
    board = pypci.lspci(vendor=vendor, device=device)
    if not board:
        raise RuntimeError(
            f"No U250 PCIe device found (vendor={vendor:#x}, device={device:#x})"
        )
    u250_bar1 = board[0].bar[1]
    return 0


def write(address, value, verify=True):
    if HARDWARE_TARGET == "u55c":
        u55c_bar2[address:address + 4] = struct.pack('<I', value & 0xffffffff)
    else:
        pypci.write(u250_bar1, address, struct.pack('<I', value))
    actual = read(address)
    if verify and actual != (value & 0xffffffff):
        raise RuntimeError(
            f"BAR write verification failed at {address:#x}: "
            f"wrote {value & 0xffffffff:#010x}, read {actual:#010x}"
        )
    return 0


def read(address):
    if HARDWARE_TARGET == "u55c":
        return struct.unpack('<I', u55c_bar2[address:address + 4])[0]
    value = pypci.read(u250_bar1, address, 4)
    return int.from_bytes(value, "little")


def commit_u55c_weights(timeout=2.0):
    write(0x100008, 1, verify=False)
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        status = read(U55C_STATUS_ADDRESS)
        if status & 0x1 and not status & 0x2:
            return
        time.sleep(0.001)
    raise TimeoutError("U55C weight commit did not complete")


# Set up connection to the influxdb database
INFLUXDB_HOST = os.getenv("DRIFTADAPT_INFLUX_HOST", "localhost")
INFLUXDB_PORT = int(os.getenv("DRIFTADAPT_INFLUX_PORT", "8086"))
INFLUXDB_DATABASE = os.getenv("DRIFTADAPT_INFLUX_DATABASE", "INFL_DB")
INFLUXDB_MEASUREMENT = os.getenv("DRIFTADAPT_INFLUX_MEASUREMENT", "basic_tcp")
labeling_window_size = int(os.getenv("DRIFTADAPT_WINDOW_SIZE", "100"))
drift_detection_threshold = float(os.getenv("DRIFTADAPT_TRIGGER_THRESHOLD", "0.20"))
total_epochs = int(os.getenv("DRIFTADAPT_TRAINING_EPOCHS", "30"))
batch_size = int(os.getenv("DRIFTADAPT_TRAINING_BATCH_SIZE", "512"))
learning_rate = float(os.getenv("DRIFTADAPT_LEARNING_RATE", "0.01"))
random_seed = int(os.getenv("DRIFTADAPT_RANDOM_SEED", "42"))
if labeling_window_size <= 0 or total_epochs <= 0 or batch_size <= 0 or learning_rate <= 0:
    raise ValueError("DRIFTADAPT window and training settings must be positive")
if not 0.0 <= drift_detection_threshold <= 1.0:
    raise ValueError("DRIFTADAPT_TRIGGER_THRESHOLD must be between 0 and 1")
torch.manual_seed(random_seed)

# Validate every file input before opening BAR1 or changing CMAC registers.
checkpoint_root = Path(__file__).resolve().parents[2] / "stimulation/checkpoint"
base_model_weights_path = Path(os.getenv(
    "DRIFTADAPT_INITIAL_MODEL", checkpoint_root / "cic-ids2017-in-network-dnn.pt"
))
labeler_dnn_weights_path = Path(os.getenv(
    "DRIFTADAPT_LABELER_MODEL", checkpoint_root / "cic-ids2017-labeler-dnn.pt"
))
for model_name, model_path, variable in (
    ("Initial model", base_model_weights_path, "DRIFTADAPT_INITIAL_MODEL"),
    ("Labeler model", labeler_dnn_weights_path, "DRIFTADAPT_LABELER_MODEL"),
):
    if not model_path.is_file():
        raise FileNotFoundError(
            f"{model_name} not found: {model_path}. "
            f"Set {variable} to its absolute path."
        )

# Deserialize and validate both state dictionaries before touching hardware.
initial_weights = torch.load(base_model_weights_path, map_location=torch.device('cpu'))
my_dnn = ID_CIC_IDS2017_small_pforest()
my_dnn.load_state_dict(initial_weights)
labeler_dnn = ID_CIC_IDS2017_large_pforest()
labeler_dnn.load_state_dict(
    torch.load(labeler_dnn_weights_path, map_location=torch.device('cpu'))
)

idbclient = InfluxDBClient(host=INFLUXDB_HOST, port=INFLUXDB_PORT, database=INFLUXDB_DATABASE)
prev_db_now = int(round(time.time() * 1000)) - 100

init(U250_VENDOR, U250_DEVICE)

# The legacy U250 design exposes CMAC enables in BAR1. U55C CMACs are managed
# by the OpenNIC driver and must not be changed through this register map.
if HARDWARE_TARGET == "u250":
    write(0x800C,1)
    write(0x8014,1)

# write initial weights to the fpga
layer_1_weights = initial_weights["linear_relu_stack.0.weight"].numpy()
layer_1_bias = initial_weights["linear_relu_stack.0.bias"].numpy()
layer_2_weights = initial_weights["linear_relu_stack.2.weight"].numpy()
layer_2_bias = initial_weights["linear_relu_stack.2.bias"].numpy()
layer_3_weights = initial_weights["linear_relu_stack.4.weight"].numpy()
layer_3_bias = initial_weights["linear_relu_stack.4.bias"].numpy()

# convert the weights into fixed point types here
all_params = np.concatenate((layer_1_weights.flatten(), layer_1_bias.flatten(), layer_2_weights.flatten(), layer_2_bias.flatten(), layer_3_weights.flatten(), layer_3_bias.flatten()))

enable_addr = int("100008",16)
base_addr = int("100010",16)

# disable register transfer flag
z = []

for y in all_params:
    z.append(c_to_fix(y))

if HARDWARE_TARGET == "u250":
    write(enable_addr,0)
for i in range(len(z)):
    write((base_addr + i*4), z[i])
    # write((base_addr + i*4), 0)
# enable register transfer flag
if HARDWARE_TARGET == "u55c":
    commit_u55c_weights()
else:
    write(enable_addr,1)

print("after writing initial weights")

# define optimizer and loss function
optimizer = torch.optim.Adam(my_dnn.parameters(), lr=learning_rate)
loss_fn = torch.nn.CrossEntropyLoss()

print("after defining data-plane dnn, optimizer, and loss function")

print("after defining labeler dnn")

# retraining related
use_generated_labels = True
use_retraining_trigger = True
my_model_f1s_proxy = []

# online training
data_buffer = []

while True:
    # add some delay here
    time.sleep(5)

    prev_db_now_temp = int(round(time.time() * 1000))
    prev_db_now_diff = prev_db_now_temp - prev_db_now
    prev_db_now = prev_db_now_temp

    # read data from the database
    results = idbclient.query(
        f'SELECT * FROM "{INFLUXDB_DATABASE}"."autogen"."{INFLUXDB_MEASUREMENT}" '
        f'WHERE time > now() - {prev_db_now_diff}ms'
    )
    points = results.get_points()

    # points_duplicate = copy.deepcopy(points)
    # print("after points = results.get_points() ", len(list(points_duplicate)))
    # print("after points = results.get_points() ", len(list(points))," ", len(data_buffer))
    # add data into the buffer list
    # flows = []
    for p in points:
        network_flow = [
            p['input1'],
            p['input2'],
            p['input3'],
            p['input4'],
            p['input5'],
            p['input6'],
            p['input7'],
            p['input8'],
            p['input9'],
            p['input10'],
            p['input11'],
            p['input12'],
            p['input13'],
            p['input14'],
            p['input15'],
            p['input16'],
            p['output'],
            p['label']
        ]
        # if None in X_point:
        #     none_count += 1
        # else:
        #     flows.append([
        #         p['src_ip'],
        #         p['dst_ip'],
        #         p['src_port'],
        #         p['dst_port'],
        #         p['proto']
        #     ])
        #     X.append(X_point)
        if None not in network_flow:
            data_buffer.append(network_flow)
        # data_buffer.append(network_flow)

    print("after data_buffer.append(network_flow) ", len(data_buffer))

    # Do online training
    while len(data_buffer) >= labeling_window_size:

        # Consume exactly one independent window. Any remainder is processed by
        # the next loop iteration rather than being mixed into this window.
        training_data = data_buffer[:labeling_window_size]
        data_buffer = data_buffer[labeling_window_size:]
        training_data_tensor = torch.tensor(training_data)

        # retrieve our prediction label and ground truth label
        ground_truth_labels = training_data_tensor[:, -1]
        prediction_labels = training_data_tensor[:, -2]

        # label data in the current window with the labeler dnn
        # _, generated_labels = labeler_dnn.run_inference(training_data_tensor)
        _, generated_labels = labeler_dnn.run_inference(training_data_tensor[:, :-2])

        if not use_generated_labels:
            generated_labels = ground_truth_labels

        # compute accuracy proxy for this window
        if use_retraining_trigger:
            f1_proxy = f1_score(
                generated_labels, prediction_labels, average="macro", zero_division=0
            )
            my_model_f1s_proxy.append(f1_proxy)

        if not use_retraining_trigger or \
            activate_retraining_trigger(f1_proxy, drift_detection_threshold):

            print("activate retraining")

            # form a dataset for retraining
            training_features, training_labels = form_balanced_binary_training_set(
                training_data_tensor[:, :-2], generated_labels
            )
            if training_features is None:
                print("skip retraining because the window does not contain both classes")
                continue
            training_dataset = OnlineDataset(training_features, training_labels)
            training_dataloader = DataLoader(training_dataset, batch_size=batch_size, shuffle=True)

            # retrain
            my_dnn, optimizer = retrain_model(my_dnn, \
                                            training_dataloader, \
                                            total_epochs, \
                                            optimizer, \
                                            loss_fn)

            # obtain the new weights
            new_state_dict = my_dnn.state_dict()

            # write the new weights back to the FPGA
            layer_1_weights = new_state_dict["linear_relu_stack.0.weight"].numpy()
            layer_1_bias = new_state_dict["linear_relu_stack.0.bias"].numpy()
            layer_2_weights = new_state_dict["linear_relu_stack.2.weight"].numpy()
            layer_2_bias = new_state_dict["linear_relu_stack.2.bias"].numpy()
            layer_3_weights = new_state_dict["linear_relu_stack.4.weight"].numpy()
            layer_3_bias = new_state_dict["linear_relu_stack.4.bias"].numpy()

            # convert the weights into fixed point types here
            all_params = np.concatenate((layer_1_weights.flatten(), layer_1_bias.flatten(), layer_2_weights.flatten(), layer_2_bias.flatten(), layer_3_weights.flatten(), layer_3_bias.flatten()))

            # Dummy Writes
            # write(0x800C,1)
            # write(0x8014,1)

            # enable_addr = int("100008",16)
            # base_addr = int("100010",16)

            start_time  = time.time()
            # disable register transfer flag

            z = []

            for y in all_params:
                z.append(c_to_fix(y))

            if HARDWARE_TARGET == "u250":
                write(enable_addr,0)
            for i in range(len(z)):
                write((base_addr + i*4), z[i])
                # write((base_addr + i*4), 0)
            # enable register transfer flag
            if HARDWARE_TARGET == "u55c":
                commit_u55c_weights()
            else:
                write(enable_addr,1)
            print("Weights Update successful")
            end_time = time.time()

            print("Time Taken:", end_time - start_time)


        # skip retraining
        else:
            print("skip retraining due to retraining trigger")
