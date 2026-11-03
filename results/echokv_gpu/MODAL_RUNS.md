# Modal GPU runs (round 6)

`scripts/echokv_bench_h100.sh` run on Modal (driver 610.57) via
`scripts/modal_bench_h100.py`. Each folder holds the per-step logs (`*.txt`)
and results (`*.json`); `gpus.txt` records the hardware actually used.

**The 8B full run was on H200s, not H100s.** These runs requested `gpu="H100"`,
which Modal may silently fill with H200 (141 GB HBM3e, ~1.4x the memory
bandwidth). The wrapper now requests `H100!`, which pins H100.

| Folder | GPUs (from gpus.txt) | Steps | Outcome |
|---|---|---|---|
| `20261101_180614/` | 1x H100 80GB HBM3 | setup unit e2e profile8b (smoke) | all passed |
| `20261101_183932/` | **4x H200** (8B on GPU 0) | all | 8B steps passed; 70B failed at startup (NCCL "unhandled cuda error" on rank 3 only, then the container was killed), see `h100_Llama-3.3-70B-Instruct/quality70b.json.*.log` |
| `20261101_213936/` | 4x H100 80GB HBM3, TP=4 | setup + all 70B steps | all passed |

8B: `20261101_183932/h100_Llama-3.1-8B-Instruct/` (H200, despite the folder
name); the only 8B H100 data is the smoke-run profile in `20261101_180614/`.
70B: `20261101_213936/h100_Llama-3.3-70B-Instruct/` (H100).
