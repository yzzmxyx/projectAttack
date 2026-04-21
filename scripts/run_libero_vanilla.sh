#!/bin/bash
set -euo pipefail

# 获取当前脚本所在目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# 切换到项目根目录，确保 experiments.* 模块导入稳定
cd "${PROJECT_ROOT}"

# ============================================================
# 结果文件夹配置
# 1. 使用当前时间生成独一无二的文件夹名字 (例如: vanilla_20260417_182700)
FOLDER_NAME="vanilla_$(date +%Y%m%d_%H%M%S)"
RESULTS_DIR="./results/${FOLDER_NAME}"

# 2. 创建该文件夹
mkdir -p "${RESULTS_DIR}"

# 3. 定义统一输出的日志文件
LOG_FILE="${RESULTS_DIR}/output.txt"

# 4. 将后续所有的控制台输出重定向到日志文件中。
# 提示: 如果你希望能同时在屏幕上看到进度条(tqdm)，你可以把下面这行替换为:
# exec > >(tee -a "${LOG_FILE}") 2>&1
exec > "${LOG_FILE}" 2>&1
# ============================================================

# export CUDA_VISIBLE_DEVICES=4
# projectAttack/experiments/robot/openvla_utils_oft.py中修改

echo "==========================================="
echo " 开始运行纯净版 LIBERO 环境测试"
echo " 结果基础目录: ${RESULTS_DIR}"
echo " 终端输出日志: ${LOG_FILE}"
echo "==========================================="

#BEST_GPU=$(nvidia-smi --query-gpu=memory.free --format=csv,nounits,noheader | \
#    cat -n | sort -k2 -rn | head -n1 | awk '{print $1-1}')

#echo "最空闲的 GPU ID 是: $BEST_GPU"

# 运行 Python 脚本（修改GPU）
# CUDA_VISIBLE_DEVICES=$BEST_GPU 
CUDA_VISIBLE_DEVICES=7 python3 "${SCRIPT_DIR}/run_libero_vanilla.py" \
    --output_dir "${RESULTS_DIR}" \

echo "==========================================="
echo " 测试完成，请查看 ${RESULTS_DIR} 文件夹。"
echo "==========================================="
