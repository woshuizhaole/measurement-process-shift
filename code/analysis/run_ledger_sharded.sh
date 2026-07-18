#!/usr/bin/env bash
# Stage 2 捷径账本: 4 分片并行 (方向×架构) 跨 4 GPU, 各 5 种子, 15 epochs, 全量
set -e
P=.
cd $P; source run/activate.sh
mkdir -p logs/analysis
run_shard () {  # gpu dir arch tag
  CUDA_VISIBLE_DEVICES=$1 nohup python -u code/analysis/20_shortcut_ledger.py \
    --seeds 5 --epochs 15 --tcap 200 --device cuda:0 \
    --dirs $2 --archs $3 --tag $4 \
    > logs/analysis/stage2_ledger_$4.log 2>&1 &
  echo "shard $4 (gpu$1 $2 $3) PID $!"
}
run_shard 0 A2B gru         A2B_gru
run_shard 1 A2B transformer A2B_tr
run_shard 2 B2A gru         B2A_gru
run_shard 3 B2A transformer B2A_tr
echo "4 shards launched"
