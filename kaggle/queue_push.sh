#!/bin/bash
SP="C:/Users/haziq/AppData/Local/Temp/claude/c--Users-haziq--vscode-Registration-PINNs/b79bbacb-d4e8-46ff-a84d-b2491ca106af/scratchpad"
QUEUE="pinn-s1-w4096-relu-max pinn-s1-w2048-relu-max pinn-s1-w1024-tanh-max pinn-s1-w1024-sin-max pinn-s1-w1024-relu-attn"
for slug in $QUEUE; do
  cd "$SP/nb_$slug" || continue
  tries=0
  while [ $tries -lt 200 ]; do
    out=$(py -3.13 -m kaggle kernels push -p . 2>&1 | tail -1)
    if echo "$out" | grep -qi "successfully pushed"; then
      echo "PUSHED $slug -> $(echo "$out" | grep -oE 'https://[^ ]*')"
      break
    elif echo "$out" | grep -qi "session count"; then
      tries=$((tries+1)); sleep 180
    else
      echo "FAILED $slug :: $out"; break
    fi
  done
done
echo "QUEUE DRAINED"
