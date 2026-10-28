#!/bin/bash
# Stage 3 push queue -- 20 notebooks, 4 conditions x 5 trunk widths.
#
# Modelled on queue2.sh: push, and if Kaggle rejects with "session count"
# (the 2-concurrent-GPU-session cap) wait 3 minutes and retry.
#
# SAFETY: this script does nothing unless you pass the literal argument GO.
#     ./queue3.sh          -> dry run, prints what it WOULD push
#     ./queue3.sh GO       -> actually pushes
#
# PREREQUISITE (blocking): the Kaggle dataset haziqahmedrumzy/pinn-sphere-conditions
# must first be updated to a version that contains toy_datasets/ and ground_truth/
# for n4_random_points, n16_random_points and n32_random_points. The version used
# by stage 1/2 has only 7 conditions and none of the new ones, so every n4/n16/n32
# notebook would fail at the DATA path. Update the dataset FIRST, then run this.

set -u
NB_ROOT="$(cd "$(dirname "$0")" && pwd)/notebooks"
PY="py -3.13"
RETRY_SLEEP=180
MAX_TRIES=200

# Longest-processing-time-first: the w4096 runs are ~1.9x the cost of the rest,
# so start them first to keep the two session slots evenly loaded at the end.
QUEUE="
s3-n4-w4096 s3-n8-w4096 s3-n16-w4096 s3-n32-w4096
s3-n4-w2048 s3-n8-w2048 s3-n16-w2048 s3-n32-w2048
s3-n4-w1024 s3-n8-w1024 s3-n16-w1024 s3-n32-w1024
s3-n4-w512  s3-n8-w512  s3-n16-w512  s3-n32-w512
s3-n4-w256  s3-n8-w256  s3-n16-w256  s3-n32-w256
"

if [ "${1:-}" != "GO" ]; then
  echo "DRY RUN -- nothing will be pushed. Pass GO to push for real."
  n=0
  for slug in $QUEUE; do
    d="$NB_ROOT/nb_$slug"
    if [ -f "$d/kernel-metadata.json" ] && [ -f "$d/nb.ipynb" ]; then
      id=$(grep -o '"id"[^,]*' "$d/kernel-metadata.json")
      echo "  OK   $slug  ($id)"
      n=$((n+1))
    else
      echo "  MISS $slug  -- $d incomplete"
    fi
  done
  echo "$n/20 notebook directories ready."
  exit 0
fi

echo "PUSHING 20 stage-3 notebooks. Ctrl-C to abort."
pushed=0; failed=0
for slug in $QUEUE; do
  d="$NB_ROOT/nb_$slug"
  if ! cd "$d" 2>/dev/null; then echo "SKIP $slug (no dir)"; failed=$((failed+1)); continue; fi
  tries=0
  while [ $tries -lt $MAX_TRIES ]; do
    out=$($PY -m kaggle kernels push -p . 2>&1 | tail -1)
    if echo "$out" | grep -qi "successfully pushed"; then
      echo "PUSHED $slug -> $(echo "$out" | grep -oE 'https://[^ ]*')"
      pushed=$((pushed+1)); break
    elif echo "$out" | grep -qi "session count"; then
      tries=$((tries+1))
      echo "  [$slug] session cap hit, retry $tries in ${RETRY_SLEEP}s"
      sleep $RETRY_SLEEP
    else
      echo "FAILED $slug :: $out"; failed=$((failed+1)); break
    fi
  done
  if [ $tries -ge $MAX_TRIES ]; then echo "GAVE UP $slug after $MAX_TRIES tries"; failed=$((failed+1)); fi
done
echo "QUEUE3 DRAINED -- pushed=$pushed failed=$failed"
