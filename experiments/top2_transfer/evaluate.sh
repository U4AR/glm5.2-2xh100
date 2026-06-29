#!/usr/bin/env bash
# Evaluate a running GLM-5.2 server: decode tok/s (median) + coherence sample.
# Usage: evaluate.sh <label>
set -uo pipefail
URL=http://localhost:8000/generate
LABEL="${1:-run}"
echo "############ EVAL: $LABEL ############"

# --- decode throughput (deterministic, temp 0) ---
P='[gMASK]<sop><|user|>\nWrite a detailed technical essay about how CPUs and GPUs differ.<|assistant|>\n'
# warmup
for i in 1 2; do curl -s "$URL" -H 'Content-Type: application/json' \
  -d "{\"text\":\"$P\",\"sampling_params\":{\"temperature\":0,\"max_new_tokens\":48}}" >/dev/null; done
echo "-- decode tok/s (5 runs x 256 new tok) --"
for i in $(seq 1 5); do
  curl -s "$URL" -H 'Content-Type: application/json' \
    -d "{\"text\":\"$P\",\"sampling_params\":{\"temperature\":0,\"max_new_tokens\":256}}" \
  | python3 -c "import sys,json;m=json.load(sys.stdin)['meta_info'];print(round(m['completion_tokens']/m['e2e_latency'],3))"
done | sort -n | awk '{a[NR]=$1} END{m=(NR%2)?a[(NR+1)/2]:(a[NR/2]+a[NR/2+1])/2; printf "  min=%.2f median=%.2f max=%.2f tok/s\n",a[1],m,a[NR]}'

# --- coherence: a reasoning + factual prompt, greedy, show the text ---
echo "-- coherence sample (greedy) --"
CP='[gMASK]<sop><|user|>\nExplain why the sky is blue, then list the first 6 prime numbers.<|assistant|>\n'
curl -s "$URL" -H 'Content-Type: application/json' \
  -d "{\"text\":\"$CP\",\"sampling_params\":{\"temperature\":0,\"max_new_tokens\":200}}" \
  | python3 -c "import sys,json;d=json.load(sys.stdin);print(d['text'][:900])"
echo
echo "-- coherence sample 2: short code --"
CP2='[gMASK]<sop><|user|>\nWrite a Python function that returns the nth Fibonacci number iteratively.<|assistant|>\n'
curl -s "$URL" -H 'Content-Type: application/json' \
  -d "{\"text\":\"$CP2\",\"sampling_params\":{\"temperature\":0,\"max_new_tokens\":200}}" \
  | python3 -c "import sys,json;d=json.load(sys.stdin);print(d['text'][:800])"
echo "############ END $LABEL ############"
