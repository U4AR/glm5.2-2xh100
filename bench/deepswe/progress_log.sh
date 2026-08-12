#!/usr/bin/env bash
# Append a one-line progress snapshot every 10 min. Costs nothing, and unlike the
# scheduled check it keeps recording if the Claude session goes away.
cd "$(dirname "$0")"
while true; do
  line="$(date +%H:%M:%S)"
  for c in $(docker ps --format '{{.Names}}' 2>/dev/null); do
    n=${c%-main-1}; n=${n%%__*}
    s=$(docker exec "$c" bash -lc 'grep -c "^Tool:" /logs/agent/mini-swe-agent.txt 2>/dev/null' 2>/dev/null)
    w=$(docker exec "$c" bash -lc 'stat -c %Y /logs/agent/mini-swe-agent.txt 2>/dev/null' 2>/dev/null)
    age=$(( $(date +%s) - ${w:-$(date +%s)} ))
    line="$line | ${n:0:14}=${s}st/${age}s"
  done
  q=$(grep -oE '#queue-req: [0-9]+' server.log 2>/dev/null | tail -1 | grep -oE '[0-9]+')
  g=$(ls jobs/top2_*/*/result.json 2>/dev/null | wc -l)
  echo "$line | queue=${q:-?} graded=$g/3" >> progress.log
  sleep 600
done
