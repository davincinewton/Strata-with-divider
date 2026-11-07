#!/usr/bin/env bash
# usage: pp-boot-test.sh <tag> <exe> [extra engine args...]
# Boots the fleet config's engine directly (the server's exact delivery: CUDA_VISIBLE_DEVICES in config order,
# config env, --serve + args + --layer-split auto), waits for the stage caches to verify ("slot 0 verified",
# which lands after the placement table and every budget ledger), then stops it and leaves the log.
set -u
cd /home/bz/Strata
TAG=$1; EXE=$2; shift 2
CFG=strata-unsloth-ud-q6_k_xl-divider-4096-3-sq.json
LOG=/tmp/pptest-$TAG.log
readarray -t SPEC < <(python3 -c "
import json
d = json.load(open('$CFG'))
print(' '.join(d['args'] + ['--layer-split', 'auto']))
print(' '.join(f'{k}={v}' for k, v in d['env'].items()))
print(','.join(str(g) for g in d['gpu']))
")
ARGS=${SPEC[0]}; ENVV=${SPEC[1]}; CVD=${SPEC[2]}
env $ENVV CUDA_VISIBLE_DEVICES=$CVD "$EXE" --serve $ARGS "$@" > "$LOG" 2>&1 &
PID=$!
for _ in $(seq 1 300); do
    grep -q "slot 0 verified" "$LOG" && break
    kill -0 $PID 2>/dev/null || break
    sleep 2
done
sleep 3
kill $PID 2>/dev/null; sleep 3; kill -9 $PID 2>/dev/null; wait $PID 2>/dev/null
echo "log: $LOG ($(wc -l < "$LOG") lines, marker: $(grep -c 'slot 0 verified' "$LOG"))"
