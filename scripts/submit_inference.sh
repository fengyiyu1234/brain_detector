#!/bin/bash
#SBATCH --job-name=brain_detector
#SBATCH --output=logs/slurm_%j.out
#SBATCH --error=logs/slurm_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=3:00:00
#SBATCH --partition=gpu
#SBATCH --qos=bme_gpu
#SBATCH --gres=gpu:1

# ── 环境 ──────────────────────────────────────────────────────────────────────
source /share/lsmsmart/fyu7/miniconda3/etc/profile.d/conda.sh
conda activate /share/lsmsmart/fyu7/miniconda3/envs/brain_detector

# ── 工作目录（脚本所在项目根目录）────────────────────────────────────────────
PROJECT_DIR="/rsstu/users/a/agrinba/DeepDesign/Fengyi/brain_detector"
cd "$PROJECT_DIR"

# ── 确保 log 目录存在 ─────────────────────────────────────────────────────────
mkdir -p logs

# ── 运行推理 ──────────────────────────────────────────────────────────────────
python scripts/run_inference.py \
    --config config/config.json
