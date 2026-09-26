# Cimba Python Tutorials

This directory follows the upstream C tutorial sequence using the 0.7 model,
input source, experiment, and result APIs. Each script is a standalone model.

The file names intentionally match the upstream C tutorial sequence:

- `hello.py` verifies the Python wrapper and native Cimba version.
- `tut_1_1.py` through `tut_1_7.py` follow the M/M/1 queue tutorial from first
  model to parallel parameter sweep.
- `tut_2_1.py`, `tut_3_1.py`, and `tut_4_1.py` contain the resource-preemption,
  amusement-park, and harbor models.
- `tut_4_0.py` is the empty harbor-model template.
- `tut_4_2.py` compares two harbor capacities with common seeds.
- `tut_5_1.py` is a three-station manufacturing-line tutorial model with
  dynamic parts, station handoffs, cycle-time, wait-time, utilization, and
  process-graph outputs.
- `multi_echelon_inventory.py` drives a six-node network with a joint
  stationary bootstrap of demand and an independent lead-time bootstrap.
- `policy_comparison.py` compares three inventory policies written as
  polymorphic `@cb.function` overrides, in one experiment, by sweeping the
  store's `policy` reference.

Run from the repository root, for example:

```bash
uv run python tutorial/tut_1_7.py -n 10 -d 1000000 -w 1000 -t
uv run python tutorial/tut_5_1.py
```
