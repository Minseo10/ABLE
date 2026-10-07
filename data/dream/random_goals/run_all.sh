#!/usr/bin/env bash
# Uniformly random goals over the person's true range (--random-goals), varied geometry,
# 4 dreams, step mode. Batch 1: the three models with 8 candidates, together. Batch 2: the
# baselines (plain, cache, the MLP with 1 candidate).
cd "$(dirname "$0")/../../.."
export OUT=data/dream/random_goals FIG=figures/dream/random_goals
MODELS="gen:flow:hist gen:ddpm:hist net:256x256:d0.3:hist" EXTRA="--random-goals" scripts/run_dream.sh generative 20 4
echo "batch 1 done $(date)"
EXTRA="--vary-geometry --random-goals" scripts/run_dream.sh fresh 20 4
echo "batch 2 done $(date)"
