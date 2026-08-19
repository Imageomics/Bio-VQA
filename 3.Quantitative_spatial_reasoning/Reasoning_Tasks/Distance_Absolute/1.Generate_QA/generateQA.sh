#!/bin/bash
#SBATCH --account=PAS2136
#SBATCH --job-name=distance-absolute
#SBATCH --nodes=1
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=12G
#SBATCH --time=08:00:00
#SBATCH --output=/users/PAS2136/rayees/Bio-VQA/1.Spatial_visual_reasoning_task/logs/Distance_Absolute_%j.out
#SBATCH --error=/users/PAS2136/rayees/Bio-VQA/1.Spatial_visual_reasoning_task/logs/Distance_Absolute_%j.err

BASE_DIR="/fs/ess/PAS2136/Rayeed/Bio-VQA/1-Spatial-Visual-Reasoning-datasets/Beetles-500"
OUT_DIR="/fs/ess/PAS2136/Rayeed/Bio-VQA/1-Spatial-Visual-Reasoning-tasks/Distance-Absolute"

/users/PAS2136/rayees/myenv/bin/python /users/PAS2136/rayees/Bio-VQA/1.Spatial_visual_reasoning_task/Distance-Absolute/generateQA.py \
    --base_dir "$BASE_DIR" \
    --output_base_dir "$OUT_DIR"