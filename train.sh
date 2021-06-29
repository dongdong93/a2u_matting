# adobe image matting
cd scripts
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 --master_port=$((RANDOM + 10000)) trainval.py \
--cfg config/adobe.yaml
