#!/bin/bash
set -e

IMG=mae
RUN="docker run --gpus all -v $(pwd):/app $IMG python mae.py"

echo "=== Building image ==="
docker build -t $IMG .

echo "=== 1/6 Pretrain ==="
$RUN pretrain

echo "=== 2/6 Plot loss curve ==="
$RUN plot_loss

echo "=== 3/6 Linear probe ==="
$RUN linear_probe --checkpoint checkpoints/mae_final.pt

echo "=== 4/6 k-NN ==="
$RUN knn --checkpoint checkpoints/mae_final.pt

echo "=== 5/6 Fine-tuning ==="
$RUN finetune --checkpoint checkpoints/mae_final.pt

echo "=== 6/6 Visualize ==="
$RUN visualize --checkpoint checkpoints/mae_final.pt

echo ""
echo "Done! Steps already computed were skipped."
echo "To force re-run, delete the corresponding output files."
