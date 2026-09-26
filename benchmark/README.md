# Benchmarks

The 0.7 M/M/1 scripts measure finite queue trials with declared distribution
inputs and object-indexed results:

```bash
uv run python benchmark/mm1.py --jobs 1000000 --reps 10
uv run python benchmark/mm1_multi.py --jobs 1000000 --trials 100
```

The first execution includes Numba class compilation. Later runs reuse the
compiled class in the same process. Use the scripts to establish new 0.7
measurements; the historical `AMD_Ryzen_7_9700X_WSL.ods` contains 0.6
results and is retained for comparison only.
