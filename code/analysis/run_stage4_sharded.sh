#!/usr/bin/env bash
# Stage 4 OPS 主实验: 按方向分 2 片跨 2 GPU, 各 3 种子 15 epoch 全量
set -e
P=.
cd $P; source run/activate.sh
rm -f results/tables/stage4_ops_main_smoke.csv results/tables/stage4_ops_ci_smoke.csv
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=6 nohup python -u code/analysis/40b_ops_gpu.py \
  --seeds 3 --epochs 15 --tcap 200 --dirs A2B --tag _A2B --device cuda:0 \
  > logs/analysis/stage4_A2B.log 2>&1 &
echo "A2B shard (gpu0) PID $!"
CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=6 nohup python -u code/analysis/40b_ops_gpu.py \
  --seeds 3 --epochs 15 --tcap 200 --dirs B2A --tag _B2A --device cuda:0 \
  > logs/analysis/stage4_B2A.log 2>&1 &
echo "B2A shard (gpu1) PID $!"
echo "2 shards launched"
