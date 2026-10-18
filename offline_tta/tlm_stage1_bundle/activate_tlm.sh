# Source this file in the remote shell to enter the project environment.
source /home/chuwenjing/miniconda3/etc/profile.d/conda.sh
conda activate /data/chuwenjing/conda-envs/tta-llm
export HF_HOME=/data/chuwenjing/cache/huggingface
export HF_DATASETS_CACHE=/data/chuwenjing/cache/huggingface/datasets
export NLTK_DATA=/data/chuwenjing/cache/nltk_data
export MPLCONFIGDIR=/data/chuwenjing/cache/matplotlib
export MPLBACKEND=Agg
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTHONPATH="/home/chuwenjing/TTA_LLM/TLM/src${PYTHONPATH:+:$PYTHONPATH}"
cd /home/chuwenjing/TTA_LLM/TLM
