#!/bin/bash
SP="C:/Users/haziq/AppData/Local/Temp/claude/c--Users-haziq--vscode-Registration-PINNs/b79bbacb-d4e8-46ff-a84d-b2491ca106af/scratchpad"
for slug in s2-w256 s2-w512 s2-dimk512 s2-dimk2048; do
  cd "$SP/nb_$slug" || continue
  t=0
  while [ $t -lt 300 ]; do
    out=$(py -3.13 -m kaggle kernels push -p . 2>&1 | tail -1)
    if echo "$out" | grep -qi "successfully pushed"; then echo "PUSHED $slug"; break
    elif echo "$out" | grep -qi "session count"; then t=$((t+1)); sleep 180
    else echo "FAILED $slug :: $out"; break; fi
  done
done
echo "QUEUE2 DRAINED"
