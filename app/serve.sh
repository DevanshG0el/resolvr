#!/usr/bin/env bash
# Start (or restart) the demo stack: FastAPI on :7400, Next.js on :7410. Logs in logs/.
set -uo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PY="${BER_PYTHON:-$HOME/ber/.venv/bin}"
export HF_HOME="${HF_HOME:-$HOME/ber/hf_cache}" BER_WORK="${BER_WORK:-$HOME/ber/work}"
mkdir -p "$ROOT/logs"
stop() { [ -f "$ROOT/logs/$1.pid" ] && kill "$(cat "$ROOT/logs/$1.pid")" 2>/dev/null; rm -f "$ROOT/logs/$1.pid"; }
case "${1:-start}" in
  stop) stop api; stop web; exit 0 ;;
esac
stop api
setsid nohup "$PY/uvicorn" server:app --app-dir "$ROOT/app" --host 127.0.0.1 --port 7400 > "$ROOT/logs/api.log" 2>&1 &
echo $! > "$ROOT/logs/api.pid"
if [ "${1:-start}" != "api" ]; then
  stop web
  (cd "$ROOT/web" && setsid nohup npx next start -p 7410 -H 127.0.0.1 > "$ROOT/logs/web.log" 2>&1 & echo $! > "$ROOT/logs/web.pid")
fi
for i in $(seq 1 40); do curl -sf localhost:7400/api/health >/dev/null && break; sleep 2; done
curl -s localhost:7400/api/health; echo
