#!/bin/bash
#SBATCH --account=PAS2136
#SBATCH --job-name=relpos-beetles
#SBATCH --nodes=1
#SBATCH --partition=batch
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=16G
#SBATCH --time=04:00:00
#SBATCH --output=/users/PAS2136/rayees/Bio-VQA/1.Spatial_visual_reasoning_task/logs/relpos_beetles_%j.out
#SBATCH --error=/users/PAS2136/rayees/Bio-VQA/1.Spatial_visual_reasoning_task/logs/relpos_beetles_%j.err

set -euo pipefail

BASE_DIR="/fs/ess/PAS2136/Rayeed/Bio-VQA/1-Spatial-Visual-Reasoning-datasets"
OUTPUT_DIR="/fs/ess/PAS2136/Rayeed/Bio-VQA/1-Spatial-Visual-Reasoning-tasks/Relative_Position_FinalV"
PYTHON="/users/PAS2136/rayees/myenv_py312/bin/python"
SCRIPT="/users/PAS2136/rayees/Bio-VQA/1.Spatial_visual_reasoning_task/Relative_Position_v2/generateQA.py"

mkdir -p /users/PAS2136/rayees/Bio-VQA/1.Spatial_visual_reasoning_task/logs
mkdir -p "$OUTPUT_DIR"

if ! grep -q -- "cyan" "$SCRIPT"; then
    echo "ERROR: $SCRIPT is an older version. Update it first." >&2
    exit 1
fi

"$PYTHON" "$SCRIPT" \
    --base_dir "$BASE_DIR" \
    --output_dir "$OUTPUT_DIR"