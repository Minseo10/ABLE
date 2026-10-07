#!/usr/bin/env bash
# The 4-dream step-mode batch one group at a time, each at the same load (5 runs):
# the cache (already launched) until it finishes, then the new model, then the deployed MLP.
cd "$(dirname "$0")/../../.."
while ps -eo args | awk '/limb_baseline[.]dream_pipeline/ && /cache_4_step_p/' | grep -q .; do sleep 60; done
echo "cache done $(date)"
ZERO=0 SEED_MODE=step MODELS=net scripts/run_dream.sh warmstart 20 4
echo "net done $(date)"
ZERO=0 SEED_MODE=step MODELS=mlp scripts/run_dream.sh warmstart 20 4
echo "mlp done $(date)"
