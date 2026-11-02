# H100 runs on Modal (round 6)

`scripts/echokv_bench_h100.sh` run on Modal H100 80GB HBM3 (driver 610.57) via
`scripts/modal_bench_h100.py`. Each folder holds the per-step logs (`*.txt`)
and results (`*.json`).

| Folder | GPUs | Steps | Outcome |
|---|---|---|---|
| `20261101_180614/` | 1x H100 | setup unit e2e profile8b (smoke) | all passed |
| `20261101_183932/` | 4x H100 (8B on GPU 0) | all | 8B steps passed; 70B failed at startup (NCCL "unhandled cuda error" on rank 3 only, then the container was killed), see `h100_Llama-3.3-70B-Instruct/quality70b.json.*.log` |
| `20261101_213936/` | 4x H100, TP=4 | setup + all 70B steps | all passed |

Use `20261101_183932/h100_Llama-3.1-8B-Instruct/` for 8B and
`20261101_213936/h100_Llama-3.3-70B-Instruct/` for 70B.
