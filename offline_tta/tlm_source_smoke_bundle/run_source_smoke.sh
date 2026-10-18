#!/usr/bin/env bash
set -euo pipefail
BUNDLE_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
TTA_CODE=/home/chuwenjing/TTA_LLM/TLM/local_repro/source_smoke
TTA_PYTHON=/data/chuwenjing/conda-envs/tta-llm/bin/python
cd "$BUNDLE_DIR"
sha256sum -c SHA256SUMS
mkdir -p "$TTA_CODE"
for name in run_source_smoke.py selected_datasets.json README.md; do
    if test -e "$TTA_CODE/$name"; then
        cmp "$BUNDLE_DIR/$name" "$TTA_CODE/$name"
    else
        cp "$BUNDLE_DIR/$name" "$TTA_CODE/$name"
    fi
done
source /home/chuwenjing/TTA_LLM/TLM/local_repro/stage1/activate_tlm.sh
export PYTHONDONTWRITEBYTECODE=1
export CUDA_VISIBLE_DEVICES="${1:-0,1}"
exec "$TTA_PYTHON" -u "$TTA_CODE/run_source_smoke.py"
