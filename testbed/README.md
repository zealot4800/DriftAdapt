# Hardware testbed

The U55C testbed runs the current simulation DriftAdapt: a distilled encoded
int8 MLP with histogram monitoring and host training from verified feedback.

See [the U55C guide](hardware/u55c/README.md) for asset export, RTL verification,
build and board operation. All commands there run from the repository root.
The simulation comparison methods remain in `simulation/`; the hardware
testbed exposes a single DriftAdapt architecture.

The traffic source replays flow feature-bin IDs from real CSV evaluation rows.
This is an internal fabric test, not a live packet classifier or a line-rate claim.
