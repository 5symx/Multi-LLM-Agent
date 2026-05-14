#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GLPFT_DIR="$(cd "${SCRIPT_DIR}/../GLPFT" && pwd)"
cd "${GLPFT_DIR}"

# BASE_DIR="/home/Multi-llm-agent"
# cd "$BASE_DIR/GLPFT"

# PLAN_PATH="/root/ymx/.cache/huggingface/hub/models--iic--alpha-umi-planner-7b"
# CAL_PATH="/root/ymx/.cache/huggingface/hub/models--iic--alpha-umi-caller-7b"
# SUM_PATH="/root/ymx/.cache/huggingface/hub/models--iic--alpha-umi-summarizer-7b"

PLAN_MODEL="${PLAN_MODEL:-/root/ymx/.cache/huggingface/hub/models--iic--alpha-umi-planner-7b}"
CAL_MODEL="${CAL_MODEL:-/root/ymx/.cache/huggingface/hub/models--iic--alpha-umi-caller-7b}"
SUM_MODEL="${SUM_MODEL:-/root/ymx/.cache/huggingface/hub/models--iic--alpha-umi-summarizer-7b}"

PLAN_PORT="${PLAN_PORT:-8001}"
CAL_PORT="${CAL_PORT:-8002}"
SUM_PORT="${SUM_PORT:-8003}"
VLLM_HOST="${VLLM_HOST:-127.0.0.1}"

PLAN_GPU="${PLAN_GPU:-0}"
CAL_GPU="${CAL_GPU:-0}"
SUM_GPU="${SUM_GPU:-1}"

LAB_DIR="${LAB_DIR:-output_res/toolbench_vllm_docker}"
P_TYPE_PLAN="${P_TYPE_PLAN:-toolbench_planner}"
P_TYPE_CAL="${P_TYPE_CAL:-toolbench_caller}"
P_TYPE_SUM="${P_TYPE_SUM:-toolbench_summarizer}"

VLLM_EXTRA_ARGS="${VLLM_EXTRA_ARGS:-}"

PLAN_LOG="${LAB_DIR}/planner_vllm.log"
CAL_LOG="${LAB_DIR}/caller_vllm.log"
SUM_LOG="${LAB_DIR}/summarizer_vllm.log"

mkdir -p "${LAB_DIR}"

cleanup() {
    if [[ -n "${PLAN_PID:-}" ]]; then kill "${PLAN_PID}" 2>/dev/null || true; fi
    if [[ -n "${CAL_PID:-}" ]]; then kill "${CAL_PID}" 2>/dev/null || true; fi
    if [[ -n "${SUM_PID:-}" ]]; then kill "${SUM_PID}" 2>/dev/null || true; fi
}
trap cleanup EXIT

wait_for_vllm() {
    local port="$1"
    local max_retry="${2:-120}"
    local i
    for ((i = 1; i <= max_retry; i++)); do
        if curl -fsS "http://${VLLM_HOST}:${port}/v1/models" >/dev/null 2>&1; then
            echo "vLLM on port ${port} is ready"
            return 0
        fi
        sleep 2
    done
    echo "vLLM on port ${port} did not become ready in time" >&2
    return 1
}

PLAN_NAME="vllm-planner"

PLAN_CID=$(
  docker run -d \
    --name "${PLAN_NAME}" \
    --runtime nvidia \
    --gpus "\"device=${PLAN_GPU}\"" \
    --ipc=host \
    -p "${PLAN_PORT}:${PLAN_PORT}" \
    -v /data0/ymx/.cache/huggingface:/root/ymx/.cache/huggingface \
    vllm/vllm-openai:latest \
    --model "${PLAN_MODEL}" \
    --port "${PLAN_PORT}" \
    --max-num-seqs 2 \
    --enforce-eager \
    --gpu-memory-utilization 0.45 \
    --served-model-name planner \
    --trust-remote-code \
    ${VLLM_EXTRA_ARGS}
)

docker logs -f "${PLAN_NAME}" > "${PLAN_LOG}" 2>&1 &
LOG_PID=$!
echo "planner container=${PLAN_CID} log_pid=${LOG_PID}"

wait_for_vllm "${PLAN_PORT}"

CAL_NAME="vllm-caller"

CAL_CID=$(
  docker run -d \
    --name "${CAL_NAME}" \
    --runtime nvidia \
    --gpus "\"device=${CAL_GPU}\"" \
    --ipc=host \
    -p "${CAL_PORT}:${CAL_PORT}" \
    -v /data0/ymx/.cache/huggingface:/root/ymx/.cache/huggingface \
    vllm/vllm-openai:latest \
    --model "${CAL_MODEL}" \
    --port "${CAL_PORT}" \
    --max-num-seqs 2 \
    --enforce-eager \
    --gpu-memory-utilization 0.45 \
    --served-model-name caller \
    --trust-remote-code \
    ${VLLM_EXTRA_ARGS}
)

docker logs -f "${CAL_NAME}" > "${CAL_LOG}" 2>&1 &
LOG_PID_CAL=$!
echo "caller container=${CAL_CID} log_pid=${LOG_PID_CAL}"

wait_for_vllm "${CAL_PORT}"


SUM_NAME="vllm-summarizer"

SUM_CID=$(
  docker run -d \
    --name "${SUM_NAME}" \
    --runtime nvidia \
    --gpus "\"device=${SUM_GPU}\"" \
    --ipc=host \
    -p "${SUM_PORT}:${SUM_PORT}" \
    -v /data0/ymx/.cache/huggingface:/root/ymx/.cache/huggingface \
    vllm/vllm-openai:latest \
    --model "${SUM_MODEL}" \
    --port "${SUM_PORT}" \
    --max-num-seqs 2 \
    --enforce-eager \
    --gpu-memory-utilization 0.45 \
    --served-model-name summarizer \
    --trust-remote-code \
    ${VLLM_EXTRA_ARGS}
)

docker logs -f "${SUM_NAME}" > "${SUM_LOG}" 2>&1 &
LOG_PID_SUM=$!
echo "summarizer container=${SUM_CID} log_pid=${LOG_PID_SUM}"

wait_for_vllm "${SUM_PORT}"