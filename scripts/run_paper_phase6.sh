#!/usr/bin/env bash
# Paper Phase 6 wrapper: int8 GSM8K-Hard column + HotpotQA 4-config sweep.
#
# Total expected wall: ~3-4 hours.
#
# Recommended tmux usage:
#   tmux new -s phase6 'bash scripts/run_paper_phase6.sh'
#   # Detach (leave it running):  Ctrl-b then d
#   # Reattach later:              tmux attach -t phase6
#   # Check progress without attaching:  tail -f /tmp/run_paper_phase6_*.log
#
# Each child script writes its own JSON to results/serving/session14_70b/...
# so a mid-run failure leaves earlier cells usable.

set -uo pipefail
cd "$(dirname "$0")/.."

STAMP=$(date -u +%Y%m%d_%H%M%S)
LOG="/tmp/run_paper_phase6_${STAMP}.log"

# Tee the wrapper's own progress to a log file too. Each child script
# writes its own per-step log under /tmp/.
exec > >(tee -a "${LOG}") 2>&1

echo "================================================================"
echo "EchoKV paper Phase 6 — multi-workload Pareto study"
echo "  start:   $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "  wrapper log: ${LOG}"
echo "  steps:"
echo "    1. static_int8 GSM8K-Hard n=200 mns=1     (~50-60 min)"
echo "    2. HotpotQA 4-config sweep n=200 mns=1    (~2-2.5 hours)"
echo "================================================================"

PHASE_RC=0

echo
echo "### Step 1/2: static_int8 GSM8K-Hard ###"
if bash scripts/01_int8_gsm8k_hard.sh; then
    echo "Step 1: OK"
else
    echo "Step 1: FAILED — continuing to step 2 anyway (independent runs)"
    PHASE_RC=1
fi

echo
echo "### Step 2/2: HotpotQA 4-config sweep ###"
if bash scripts/02_hotpotqa_sweep.sh; then
    echo "Step 2: OK"
else
    echo "Step 2: FAILED"
    PHASE_RC=1
fi

echo
echo "================================================================"
echo "Phase 6 wrapper complete: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
[[ ${PHASE_RC} -eq 0 ]] && echo "All steps OK." || echo "One or more steps failed (rc=${PHASE_RC}). See per-step logs."
echo "================================================================"

exit ${PHASE_RC}
