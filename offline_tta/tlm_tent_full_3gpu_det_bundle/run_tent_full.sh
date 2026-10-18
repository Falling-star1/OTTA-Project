#!/usr/bin/env bash
set -euo pipefail
TTA_BUNDLE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
TTA_CODE=/home/chuwenjing/TTA_LLM/TLM/local_repro/tent_full_3gpu_det
TTA_PYTHON=/data/chuwenjing/conda-envs/tta-llm/bin/python
TTA_ACTION="${1:-preflight}"
TTA_GPUS="${2:-0,1,2}"
TTA_OUTPUT="${3:-}"
case "$TTA_ACTION" in preflight|run|resume) ;; *) echo 'Action must be preflight, run, or resume.' >&2; exit 2 ;; esac
if [[ ! "$TTA_GPUS" =~ ^[0-9]+,[0-9]+,[0-9]+$ ]]; then
    echo 'Specify three distinct physical GPU IDs, e.g. 0,1,2.' >&2
    exit 2
fi
if [[ "$TTA_ACTION" == resume && -z "$TTA_OUTPUT" ]] || [[ "$TTA_ACTION" != resume && -n "$TTA_OUTPUT" ]] || [[ $# -gt 3 ]]; then
    echo 'Usage: bash run_tent_full.sh preflight|run GPU0,GPU1,GPU2; or resume GPU0,GPU1,GPU2 EXACT_OUTPUT_DIR' >&2
    exit 2
fi
cd "$TTA_BUNDLE"
command -v flock >/dev/null
exec 9>"$TTA_BUNDLE/.run.lock"
flock -n 9 || { echo 'This full bundle already has a running job.' >&2; exit 3; }
sha256sum -c SHA256SUMS
mkdir -p "$TTA_CODE"
for name in config.json selected_datasets.json tent_text.py test_tent_text.py offline_flow.py full_core.py test_full_core.py job_state.py test_job_state.py full_job.py test_full_job.py full_runtime.py run_dataset.py run_tent_full.py test_controller.py gpu_layout.py test_gpu_layout.py reproducibility.py test_reproducibility.py README.md DESIGN.md; do
    if test -e "$TTA_CODE/$name"; then
        cmp "$TTA_BUNDLE/$name" "$TTA_CODE/$name"
    else
        cp "$TTA_BUNDLE/$name" "$TTA_CODE/$name"
    fi
done
source /home/chuwenjing/TTA_LLM/TLM/local_repro/stage1/activate_tlm.sh
export PYTHONDONTWRITEBYTECODE=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8
cd "$TTA_CODE"
"$TTA_PYTHON" -c 'import sys; from gpu_layout import physical_devices; physical_devices(sys.argv[1], 3)' "$TTA_GPUS"
CUDA_VISIBLE_DEVICES="" "$TTA_PYTHON" -u -m unittest discover -s "$TTA_CODE" -p 'test_*.py' -v
export CUDA_VISIBLE_DEVICES="$TTA_GPUS"
if [[ "$TTA_ACTION" == resume ]]; then
    exec "$TTA_PYTHON" -u "$TTA_CODE/run_tent_full.py" resume --output "$TTA_OUTPUT"
else
    exec "$TTA_PYTHON" -u "$TTA_CODE/run_tent_full.py" "$TTA_ACTION"
fi
