source /mnt/shared-storage-user/chengshuang/anaconda3/etc/profile.d/conda.sh

conda activate llamafactory_dllm

export HF_ENDPOINT="https://hf-mirror.com"
export HF_HOME="/mnt/shared-storage-user/chengshuang/.hf_home"
export MODELSCOPE_CACHE="/mnt/shared-storage-user/chengshuang/.cache/modelscope"
export FAIRSEQ2_CACHE_DIR="/mnt/shared-storage-user/chengshuang/.cache/fairseq2"
export TORCH_CUDA_CACHE_PATH="/mnt/shared-storage-user/chengshuang/.cache/torch"
export COMPASS_DATA_CACHE="/mnt/shared-storage-user/chengshuang/.cache/compass"

export HTTP_PROXY=http://liudawei:HuDjhMeoJxdKJATO2ljtNqmbIoL2MAkKRcgTXi1nXZ5CeUKPXR77MWOOVyG2@10.1.20.50:23128
export HTTPS_PROXY=http://liudawei:HuDjhMeoJxdKJATO2ljtNqmbIoL2MAkKRcgTXi1nXZ5CeUKPXR77MWOOVyG2@10.1.20.50:23128
export no_proxy="hf-mirror.com,$no_proxy"  # 大多数工具（如 curl、wget）
export NO_PROXY="hf-mirror.com,$NO_PROXY"  # 部分工具（如 Python 的 requests）
export WANDB_BASE_URL=https://api.bandw.top
export WANDB_API_KEY=609ee35b356dfc8afb95c98599838108995fd7e5
export WANDB_PROJECT="dmllm"
export WANDB_ENTITY="chengs18"
export TORCH_DISTRIBUTED_DEBUG=INFO
export NCCL_TIMEOUT=3600000
wandb offline

cd /mnt/shared-storage-user/chengshuang/projects/SDAR/training/llama_factory_sdar

NODE_COUNT=${NODE_COUNT:-1}
NODE_RANK=${NODE_RANK:-0}
NPROC_PER_NODE=${PROC_PER_NODE:-8}
MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
MASTER_PORT=${MASTER_PORT:-6000}

torchrun \
    --nnodes ${NODE_COUNT} \
    --node_rank ${NODE_RANK} \
    --nproc_per_node ${NPROC_PER_NODE} \
    --master_addr ${MASTER_ADDR} \
    --master_port ${MASTER_PORT} \
    ./src/llamafactory/launcher.py \
    ./examples/train_full_sdar/sdar_4b/sdar_8b_math_cot_full.yaml
