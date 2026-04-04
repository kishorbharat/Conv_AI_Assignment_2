#!/bin/bash
echo "Waiting for training to complete..."
while ps aux | grep "run_rag_terminal" | grep -v grep > /dev/null; do
  sleep 30
done
echo "Training finished. Committing changes..."
cd /workspaces/Conv_AI_Assignment_2
git add -A
git commit -m "Save trained checkpoint after completed training run"
git push origin main
echo "Done. Changes committed and pushed."
