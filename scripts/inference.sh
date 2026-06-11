#!/bin/bash
#BSUB -J brain_detector_post
#BSUB -o logs/lsf_%J.out
#BSUB -e logs/lsf_%J.err
#BSUB -n 4
#BSUB -R "rusage[mem=32]"
#BSUB -W 48:00
#BSUB -q bme_gpu
#BSUB -gpu "num=2:mps=no:j_exclusive=yes"

# ── 环境 ──────────────────────────────────────────────────────────────────────
source /share/lsmsmart/fyu7/miniconda3/etc/profile.d/conda.sh
conda activate /share/lsmsmart/fyu7/miniconda3/envs/brain_detector

# ── 工作目录 ──────────────────────────────────────────────────────────────────
PROJECT_DIR="/rsstu/users/a/agrinba/DeepDesign/Fengyi/brain_detector"
cd "$PROJECT_DIR"

# ── 确保 log 目录存在 ─────────────────────────────────────────────────────────
mkdir -p logs

# ── 验证 GPU/CUDA 可用（驱动不兼容时立即失败，避免静默回退到 CPU）────────────
echo "=== GPU 环境检查 ==="
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
python - <<'EOF'
import torch, sys
if not torch.cuda.is_available():
    print(f"[ERROR] CUDA 不可用！节点: $(hostname)，请换节点重新提交。", flush=True)
    sys.exit(1)
print(f"[OK] CUDA 可用: {torch.cuda.get_device_name(0)}, 驱动版本: {torch.version.cuda}", flush=True)
EOF
if [ $? -ne 0 ]; then
    exit 1
fi
echo "===================="

# ── 运行推理 ──────────────────────────────────────────────────────────────────
python scripts/run_inference.py \
    --config config/config.json

#config_PreAlign.json
#config.json