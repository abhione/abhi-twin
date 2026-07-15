#!/usr/bin/env bash
# v1.5 reproducibility gate: clean Spark -> `git clone && make video-demo` ->
# working MuseTalk demo in < 30 min (hard constraint #5). Host: spark.
set -euo pipefail

if [[ "$(uname -s)" != "Linux" ]]; then
  echo "=== RUN ON SPARK === make video-demo runs the MuseTalk port on the Spark." >&2
  exit 2
fi

START=$(date +%s)
cd "$(dirname "$0")/../.."

echo "==> [1/4] building + starting the musetalk service (first build is the long pole)"
docker compose -f docker/compose.yaml --profile video up -d --build musetalk

echo "==> [2/4] waiting for the service"
for _ in $(seq 1 60); do
  curl -fsS http://localhost:8004/healthz >/dev/null 2>&1 && break
  sleep 5
done
curl -fsS http://localhost:8004/healthz

echo "==> [3/4] stock sanity inference (Obama sample, addendum Step 3)"
docker compose -f docker/compose.yaml exec musetalk \
  python -m scripts.inference --inference_config configs/inference/test.yaml \
  --result_dir /twin/logs/musetalk-demo

echo "==> [4/4] benchmark gate (FPS >= 10 @256 fails the build if unmet)"
.venv/bin/python eval/video.py --gate || python3 eval/video.py --gate

ELAPSED=$(( ($(date +%s) - START) / 60 ))
echo "=== video-demo complete in ${ELAPSED} min (reproducibility bar: 30 min) ==="
echo "Output frames: /twin/logs/musetalk-demo — fill the table in docs/PUBLICATION.md"
