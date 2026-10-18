#!/usr/bin/env bash
set -euo pipefail
BUNDLE_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
TTA_PYTHON=/data/chuwenjing/conda-envs/tta-llm/bin/python
TTA_CODE=/home/chuwenjing/TTA_LLM/TLM/local_repro/stage1
cd "$BUNDLE_DIR"
sha256sum -c SHA256SUMS
test -x "$TTA_PYTHON"
mkdir -p "$TTA_CODE"
for name in prepare_data.py selected_datasets.json PROTOCOL.md activate_tlm.sh; do
    if test -e "$TTA_CODE/$name"; then
        cmp "$BUNDLE_DIR/$name" "$TTA_CODE/$name"
    else
        cp "$BUNDLE_DIR/$name" "$TTA_CODE/$name"
    fi
done
export PYTHONDONTWRITEBYTECODE=1
"$TTA_PYTHON" "$TTA_CODE/prepare_data.py" "$BUNDLE_DIR"
