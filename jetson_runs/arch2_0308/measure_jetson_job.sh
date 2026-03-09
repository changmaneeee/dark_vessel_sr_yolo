#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 5 ]]; then
  echo "usage: measure_jetson_job.sh <job_name> <run_dir> <metrics_json_path> <tegrastats_interval_ms> -- <command...>" >&2
  exit 1
fi

JOB_NAME="$1"; shift
RUN_DIR="$1"; shift
METRICS_JSON="$1"; shift
TEGR_INTERVAL_MS="$1"; shift

if [[ "$1" != "--" ]]; then
  echo "Expected -- before command" >&2
  exit 1
fi
shift

mkdir -p "$RUN_DIR"
CMD_LOG="$RUN_DIR/${JOB_NAME}.stdout.log"
TGR_LOG="$RUN_DIR/${JOB_NAME}.tegrastats.log"
TIME_LOG="$RUN_DIR/${JOB_NAME}.time.log"
SUMMARY_JSON="$RUN_DIR/${JOB_NAME}.summary.json"
SUMMARY_TXT="$RUN_DIR/${JOB_NAME}.summary.txt"

printf '%q ' "$@" > "$RUN_DIR/${JOB_NAME}.command.txt"
printf '\n' >> "$RUN_DIR/${JOB_NAME}.command.txt"

cleanup() {
  if [[ -n "${TG_PID:-}" ]] && ps -p "$TG_PID" >/dev/null 2>&1; then
    kill "$TG_PID" >/dev/null 2>&1 || true
    sleep 1
    kill -9 "$TG_PID" >/dev/null 2>&1 || true
  fi
}
trap cleanup EXIT

if command -v tegrastats >/dev/null 2>&1; then
  tegrastats --interval "$TEGR_INTERVAL_MS" > "$TGR_LOG" 2>&1 &
  TG_PID=$!
else
  echo "[WARN] tegrastats not found. Power summary will be missing." | tee "$CMD_LOG"
fi

RC=0
/usr/bin/time -v -o "$TIME_LOG" "$@" >> "$CMD_LOG" 2>&1 || RC=$?
cleanup
trap - EXIT

if [[ $RC -ne 0 ]]; then
  echo "[ERROR] job failed with exit code $RC. See $CMD_LOG" >&2
  exit $RC
fi

if [[ -f "$METRICS_JSON" ]]; then
  python3 "$(dirname "$0")/jetson_job_summary.py" "$JOB_NAME" "$METRICS_JSON" "$TGR_LOG" "$SUMMARY_JSON" > "$SUMMARY_TXT"
else
  echo "[WARN] metrics json not found: $METRICS_JSON" | tee "$SUMMARY_TXT"
fi
