#!/usr/bin/env bash

set -euo pipefail

# SCRIPT="tabular_trainer.py"
# SCRIPT="vision_trainer.py"
# SCRIPT="v_trainer.py"
SCRIPT="multimodal_trainer.py"

echo "============================================================"
echo "Running all ADNI Tabular experiments"
echo "============================================================"

declare -a EXPERIMENTS=(
    "0:Multiclass (CN vs MCI vs AD)"
    "1:CN vs AD"
    "2:CN vs MCI"
    "3:MCI vs AD"
)

for exp in "${EXPERIMENTS[@]}"; do
    IFS=":" read -r id name <<< "$exp"

    echo
    echo "============================================================"
    echo "Experiment $id - $name"
    echo "============================================================"

    python "$SCRIPT" --experiment "$id"

    echo "✓ Finished Experiment $id"
done

echo
echo "============================================================"
echo "All experiments completed successfully."
echo "============================================================"