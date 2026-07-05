# Cimba Python Tutorials

This directory mirrors `subprojects/cimba/tutorial/` with Python versions built
on `cimba.sim`.

The file names intentionally match the upstream C tutorial sequence:

- `hello.py` verifies the Python wrapper and native Cimba version.
- `tut_1_1.py` through `tut_1_7.py` follow the M/M/1 queue tutorial from first
  model to parallel parameter sweep.
- `tut_2_1.py`, `tut_3_1.py`, and `tut_4_1.py` point at the larger Python demo
  ports already maintained under `examples/`.
- `tut_4_0.py` is the empty harbor-model template.
- `tut_4_2.py` sketches the harbor experiment sweep on top of the Python harbor
  model.
- `tut_5_1.py` records that the CUDA/GPU tutorial is not exposed in the Python
  API yet.
- `assembly_line.py` is a standalone three-station manufacturing-line tutorial
  model with cycle-time, wait-time, utilization, and process-graph outputs.

Run from the repository root, for example:

```bash
uv run python tutorial/tut_1_7.py -n 10 -d 1000000 -w 1000 -t
uv run --extra plot python tutorial/assembly_line.py
```
