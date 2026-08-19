# DRIFTADAPT Alveo U55C port

This port targets the detected `xcu55c-fsvh2892-2L-e` using the existing
Vivado 2022.2 OpenNIC checkout. It does not modify or program the FTC image.

The runtime path is host PF0 -> DRIFTADAPT DNN -> QSFP0, then QSFP1 -> host PF1.
The external device between QSFP0 and QSFP1 must return the Ethernet frame
unchanged and preserve bytes 14 through 48. A direct QSFP0-to-QSFP1 loop is
also valid. Both links use 100GbE CAUI-4 with integrated RS-FEC.

The DNN shares one registered 32-bit multiplier across its 168 MACs. A valid
single-beat feature packet therefore takes 672 cycles (2.688 microseconds at
250 MHz). This is intentionally optimized for timing-safe experiment traffic,
not minimum-size 100GbE line rate.

The BAR2 user block begins at `0x100000`: feature word `DRFT` at `+0x000`,
status at `+0x004`, commit at `+0x008`, 182 Q16.16 parameters at `+0x010`,
and classified/bypassed packet counters at `+0x300/+0x308`.

Create the project without programming hardware:

```bash
bash scripts/build_u55c.sh create
```

Run synthesis, implementation, timing closure, routing, DRC, and bitstream
generation with one command:

```bash
bash scripts/build_u55c.sh all
```

If implementation already produced the routed checkpoint and only the closure
step needs to be resumed, use:

```bash
bash scripts/build_u55c.sh close
```

Programming is intentionally separate and is allowed only after the complete
implementation gate passes and the current FPGA image has been preserved.

After a passing build, inspect the exact image and host prerequisites without
changing hardware:

```bash
bash scripts/bringup_u55c.sh --dry-run
```

The real operation requires an interactive `PROGRAM DRIFTADAPT U55C`
confirmation:

```bash
bash scripts/bringup_u55c.sh
```
