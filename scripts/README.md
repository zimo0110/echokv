# EchoKV experiment scripts

Self-contained `bash` scripts for the Phase-6 multi-workload Pareto
study. Designed to survive SSH disconnects via `tmux`.

## Files

| script | what it runs | wall time |
|--|--|--|
| `00_smoke_hotpotqa.sh` | 4-prompt FP16 sanity on Llama-3.1-8B + TP=2 | ~2 min |
| `01_int8_gsm8k_hard.sh` | static_int8 GSM8K-Hard n=200 mns=1 | ~50-60 min |
| `02_hotpotqa_sweep.sh` | HotpotQA 4 configs (FP16/INT4/INT8/EchoKV-3C) n=200 mns=1 | ~2-2.5 hours |
| `run_paper_phase6.sh` | wrapper: 01 → 02 sequentially | ~3-4 hours |

Each child script:

- Sets `CUDA_VISIBLE_DEVICES=4,5,6,7` (override by exporting the var
  before invoking).
- Sets `VLLM_ALLOW_INSECURE_SERIALIZATION=1` and
  `FLASHINFER_DISABLE_VERSION_CHECK=1`.
- Logs to `/tmp/<script>_<utc-stamp>.log`.
- Writes the result JSON to `results/serving/session14_70b/...`.
- Each cell writes its own JSON, so a mid-sweep failure leaves
  earlier cells usable.

## Recommended invocation (tmux for SSH durability)

```bash
# Smoke test first — catches any environment/import bugs before
# committing 70B time:
tmux new -s smoke -d 'bash scripts/00_smoke_hotpotqa.sh'
tmux attach -t smoke      # watch live; detach with Ctrl-b then d
# When smoke is done and the JSON looks reasonable:
tmux kill-session -t smoke

# Then the main sweep — detached, persists across SSH drop:
tmux new -s phase6 -d 'bash scripts/run_paper_phase6.sh'
# Detach: already detached because of -d. Reattach to watch:
tmux attach -t phase6
# Quick progress check without attaching:
tail -f /tmp/run_paper_phase6_*.log
```

## Resuming after a partial failure

The two main steps write independent JSONs:

- Step 1 → `results/serving/session14_70b/publication/int8_n200_seq1.json`
- Step 2 → `results/serving/session14_70b/hotpotqa/{fp16,int4,int8,echokv3c_*}_n200_seq1.json`

If step 1 succeeded but step 2 failed mid-way, just re-run step 2:

```bash
tmux new -s hotpot -d 'bash scripts/02_hotpotqa_sweep.sh'
```

The sweep script always re-runs all 4 cells; it doesn't currently
skip cells whose JSON already exists. Either move the existing
JSONs aside or accept the re-run cost.

## Cleanup if a run is killed

A killed driver process can leave 4 zombie `VLLM::Worker_TP*`
processes holding ~70 GB each on GPUs 4-7. To recover:

```bash
# Find your zombies (filter to your user, ignore other users' jobs):
ps -ef | grep -E "Worker_TP|EngineCore" | grep "$(whoami)" | grep -v grep
# Kill them by PID. Use SIGKILL if SIGTERM doesn't take:
kill -9 <pids>
# Verify GPUs 4-7 are clean:
nvidia-smi --query-gpu=index,memory.used --format=csv | sed -n '1p;6,9p'
```

## Output paths

Results land under `results/serving/session14_70b/`:

```
publication/
    int8_n200_seq1.json                  ← step 1
    fp16_n200_seq1.json                  ← from earlier sessions
    int4_n200_seq1.json                  ← from earlier sessions
    echokv3c_abscanary_tau022_n200_seq1.json   ← from Phase 5

hotpotqa/                                 ← all step-2 cells
    fp16_n200_seq1.json
    int4_n200_seq1.json
    int8_n200_seq1.json
    echokv3c_abscanary_tau022_n200_seq1.json
```
