#!/bin/bash

# Create logs directory if it doesn't exist
mkdir -p logs

trap 'echo "Caught interrupt signal. Killing all fold processes..."; kill 0; exit 1' INT

echo "🚀 Launching nested CV folds..."

for FOLD in 1 2 3 4 5; do
    echo "➡️  Launching Fold $FOLD..."
    python nested_cv_warfarinnet.py --fold $FOLD > logs/real_fold_net_${FOLD}.log 2>&1 &
    PIDS[$FOLD]=$!
done

# Wait for all background jobs
wait

echo "All folds completed (or terminated). Check logs in ./logs"
