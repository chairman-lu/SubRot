export HF_HOME=/home/disk/workspace/hf_download/
export HF_DATASETS_CACHE=/home/disk/workspace/hf_download/datasets
export LMUData=/home/disk/workspace_hz/LMUData
set -e

# get subspace
python -m vlmrot.subspaceloss --config configs/llava-next-7b-subrot-W4A4.yaml

# calibration
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nnodes=1 --nproc_per_node=4 --master_addr=127.0.0.1 --master_port=29501 optimize_subrot.py --config configs/llava-next-7b-subrot-W4A4.yaml

# eval
python -W ignore main.py --config configs/llava-next-7b-subrot-W4A4.yaml