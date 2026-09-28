#!/usr/bin/env bash
# One intermediate report of the rerun: figures named by the time of the check, and
# the summary table next to them. Figures: figures/dream/rerun/interim/interim_HHMM*.png
cd "$(dirname "$0")/../../.."
export PATH="$HOME/anaconda3/envs/kinder/bin:$PATH"
stamp=$(date +%H%M)
csvs=$(for f in data/dream/rerun/*_p*.csv; do case $f in *_dreams.csv) ;; *) [ "$(wc -l < "$f")" -gt 1 ] && echo "$f";; esac; done)
[ -z "$csvs" ] && { echo "no rows yet at $stamp"; exit 0; }
out=figures/dream/rerun/interim/interim_$stamp.png
python scripts/make_dream_figures.py --csv $csvs --out "$out" --per-person --max-episodes 10 \
    --title "Interim at $(date +%H:%M) (runs still going); band = standard error" \
    > "figures/dream/rerun/interim/interim_$stamp.txt" 2>&1
echo "interim $stamp: $(grep -c '^Wrote' figures/dream/rerun/interim/interim_$stamp.txt) figures, exit $?"
