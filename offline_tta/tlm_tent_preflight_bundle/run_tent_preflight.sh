#!/usr/bin/env bash
set -euo pipefail
TTA_BUNDLE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
TTA_CODE=/home/chuwenjing/TTA_LLM/TLM/local_repro/tent_preflight
TTA_PYTHON=/data/chuwenjing/conda-envs/tta-llm/bin/python
TTA_GPUS="${1:-0,1}"
if [[ ! "$TTA_GPUS" =~ ^[0-9]+,[0-9]+$ ]] || [[ "${TTA_GPUS%,*}" == "${TTA_GPUS#*,}" ]]; then
    echo "Choose two distinct physical GPU IDs, for example: bash run_tent_preflight.sh 0,1" >&2
    exit 2
fi
cd "$TTA_BUNDLE"
command -v flock >/dev/null
exec 9>"$TTA_BUNDLE/.run.lock"
flock -n 9 || { echo 'A Tent preflight is already running; do not start another.' >&2; exit 3; }
sha256sum -c SHA256SUMS
mkdir -p "$TTA_CODE"
for name in tent_text.py test_tent_text.py run_tent_preflight.py config.json selected_datasets.json README.md DESIGN.md; do
    if test -e "$TTA_CODE/$name"; then
        cmp "$TTA_BUNDLE/$name" "$TTA_CODE/$name"
    else
        cp "$TTA_BUNDLE/$name" "$TTA_CODE/$name"
    fi
done
source /home/chuwenjing/TTA_LLM/TLM/local_repro/stage1/activate_tlm.sh
export PYTHONDONTWRITEBYTECODE=1
cd "$TTA_CODE"
# Run the small random model checks on CPU before allocating the real Qwen model.
CUDA_VISIBLE_DEVICES="" "$TTA_PYTHON" -u -m unittest discover -s "$TTA_CODE" -p test_tent_text.py -v
export CUDA_VISIBLE_DEVICES="$TTA_GPUS"
exec "$TTA_PYTHON" -u "$TTA_CODE/run_tent_preflight.py"
