#!/usr/bin/env bash
# Repeatable single-stream decode benchmark. Reports median tok/s from meta_info.
set -uo pipefail
URL=http://localhost:8000/generate
RUNS=${1:-5}; TOK=${2:-256}
P='[gMASK]<sop><|user|>\nWrite a detailed technical essay about how CPUs and GPUs differ.<|assistant|>\n'
for i in 1 2 3; do curl -s "$URL" -H 'Content-Type: application/json' -d "{\"text\":\"$P\",\"sampling_params\":{\"temperature\":0,\"max_new_tokens\":48}}" >/dev/null; done
for i in $(seq 1 "$RUNS"); do
  curl -s "$URL" -H 'Content-Type: application/json' -d "{\"text\":\"$P\",\"sampling_params\":{\"temperature\":0,\"max_new_tokens\":$TOK}}" \
   | python3 -c "import sys,json;m=json.load(sys.stdin)['meta_info'];print(round(m['completion_tokens']/m['e2e_latency'],3))"
done | sort -n | awk '{a[NR]=$1} END{if(NR==0){print "no samples";exit 1} m=(NR%2)?a[(NR+1)/2]:(a[NR/2]+a[NR/2+1])/2; printf "runs=%d min=%.2f median=%.2f max=%.2f tok/s (incl prefill, amortized)\n",NR,a[1],m,a[NR]}'
