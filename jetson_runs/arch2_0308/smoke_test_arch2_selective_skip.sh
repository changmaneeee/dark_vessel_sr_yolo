#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ARCH2_PY="${ARCH2_PY:-$SCRIPT_DIR/arch2_softgate_selective_skip.py}"

python -m py_compile \
  "$ARCH2_PY" \
  "$SCRIPT_DIR/arch2_bench_selective_skip.py" \
  "$SCRIPT_DIR/test_arch2_selective_skip_unit.py"

python "$SCRIPT_DIR/test_arch2_selective_skip_unit.py" --arch2_py "$ARCH2_PY"

echo "[DONE] Arch2 selective-skip smoke test passed."
