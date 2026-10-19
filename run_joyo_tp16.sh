#!/bin/bash
# JoyOV2 T2VA TP=EP=16 常驻 + TE TP16 常驻 + VAE 常驻（无 rolling）
# 适配自刘朋 2026-10-09 部署脚本（98 服务器 / xllm_wh 容器路径）。
# 需要: 16 张空闲 NPU、已编译的 xllm、Ascend 环境
#
# 用法: bash run_joyo_tp16.sh cpp|python
#   cpp    - C++ Libtorch 组图（原生管线，精度 golden）
#   python - PyTorch 组图（--model_impl python + --python_model_path 指向仓库根）

set -euo pipefail

MODE="${1:?usage: run_joyo_tp16.sh cpp|python}"
if [ "${MODE}" != "cpp" ] && [ "${MODE}" != "python" ]; then
  echo "usage: run_joyo_tp16.sh cpp|python" >&2
  exit 2
fi

REPO=/export/home/ext.xumaojun1/wanghao/xllm
XLLM_PATH=${REPO}/build/lib.linux-aarch64-cpython-311/xllm/xllm
MODEL_PATH=/mnt/cfs/9n-das-admin/llm_models/joyo_v2_diffusers_stage3p5_it526400
MASTER_NODE_ADDR=127.0.0.1:17600
START_PORT=18500
START_DEVICE=0
NNODES=16
TP_SIZE=16
LOG_DIR=${REPO}/joyai-video/logs/run_tp16_${MODE}
mkdir -p "${LOG_DIR}"

test -x "${XLLM_PATH}" || { echo "xllm binary not found: ${XLLM_PATH}" >&2; exit 1; }
test -d "${MODEL_PATH}" || { echo "model path not found: ${MODEL_PATH}" >&2; exit 1; }

set +u
source /usr/local/Ascend/ascend-toolkit/set_env.sh
source /usr/local/Ascend/nnal/atb/set_env.sh
set -u

export ASDOPS_LOG_TO_STDOUT=1
export ASDOPS_LOG_LEVEL=ERROR
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export NPU_MEMORY_FRACTION=1.0
export HCCL_IF_BASE_PORT=43440

EXTRA_FLAGS=()
if [ "${MODE}" = "python" ]; then
  EXTRA_FLAGS+=(--model_impl=python "--python_model_path=${REPO}")
fi

# 清上次 SHM，避免 size 不兼容
rm -f /dev/shm/xllm_17600_* 2>/dev/null || true

for (( i=0; i<NNODES; i++ )); do
  PORT=$((START_PORT + i))
  DEV=$((START_DEVICE + i))
  ASCEND_RT_VISIBLE_DEVICES="${DEV}" \
    "${XLLM_PATH}" \
    --model="${MODEL_PATH}" \
    --backend=dit \
    --npu_kernel_backend=TORCH \
    --max_memory_utilization=0.8 \
    --tp_size="${TP_SIZE}" \
    --cfg_size=1 \
    --sp_size=1 \
    --vae_size=1 \
    --dit_enable_joyo_text_encoder=true \
    --text_encoder_tp_size=16 \
    --master_node_addr="${MASTER_NODE_ADDR}" \
    --nnodes="${NNODES}" \
    --port="${PORT}" \
    --communication_backend=hccl \
    --enable_prefix_cache=false \
    --enable_chunked_prefill=false \
    --enable_schedule_overlap=false \
    --enable_shm=true \
    --input_shm_size=1024 \
    --output_shm_size=1024 \
    --node_rank="${i}" \
    ${EXTRA_FLAGS[@]+"${EXTRA_FLAGS[@]}"} \
    > "${LOG_DIR}/node_${i}.log" 2>&1 &
  echo "rank=${i} pid=$! port=${PORT} device=${DEV}"
done

echo "MODE=${MODE} LOG_DIR=${LOG_DIR}"
echo "wait: grep -i 'Brpc Server started' ${LOG_DIR}/node_0.log"
