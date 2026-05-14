#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GLPFT_DIR="$(cd "${SCRIPT_DIR}/../GLPFT" && pwd)"
cd "${GLPFT_DIR}"

PLAN_MODEL="${PLAN_MODEL:-models--iic--alpha-umi-planner-7b.gguf}"
CAL_MODEL="${CAL_MODEL:-models--iic--alpha-umi-caller-7b.gguf}"
SUM_MODEL="${SUM_MODEL:-models--iic--alpha-umi-summarizer-7b.gguf}"

PLAN_NAME="${PLAN_NAME:-planner}"
CAL_NAME="${CAL_NAME:-caller}"
SUM_NAME="${SUM_NAME:-summarizer}"

MODELS_DIR="${MODELS_DIR:-/data0/ymx/cache/llama.cpp/models--iic--alpha-umi-GGUF/}"

PLAN_PORT="${PLAN_PORT:-8001}"
CAL_PORT="${CAL_PORT:-8002}"
SUM_PORT="${SUM_PORT:-8003}"
LLAMA_HOST="${LLAMA_HOST:-127.0.0.1}"

PLAN_GPU="${PLAN_GPU:-0}"
CAL_GPU="${CAL_GPU:-0}"
SUM_GPU="${SUM_GPU:-1}"

CTX_SIZE="${CTX_SIZE:-4096}"
N_GPU_LAYERS="${N_GPU_LAYERS:-1}"
N_THREADS="${N_THREADS:-2}"
N_BATCH="${N_BATCH:-512}"

LAB_DIR="${LAB_DIR:-output_res/toolbench_llama_docker}"
P_TYPE_PLAN="${P_TYPE_PLAN:-toolbench_planner}"
P_TYPE_CAL="${P_TYPE_CAL:-toolbench_caller}"
P_TYPE_SUM="${P_TYPE_SUM:-toolbench_summarizer}"

PLAN_LOG="${LAB_DIR}/planner_llama.log"
CAL_LOG="${LAB_DIR}/caller_llama.log"
SUM_LOG="${LAB_DIR}/summarizer_llama.log"

# mkdir -p "${LAB_DIR}"

# cleanup() {
#     docker rm -f llama-planner llama-caller llama-summarizer >/dev/null 2>&1 || true
# }
# trap cleanup EXIT

# wait_for_llama_server() {
#     local port="$1"
#     local max_retry="${2:-120}"
#     local i
#     for ((i = 1; i <= max_retry; i++)); do
#         if curl -fsS "http://${LLAMA_HOST}:${port}/v1/models" >/dev/null 2>&1; then
#             echo "llama-server on port ${port} is ready"
#             return 0
#         fi
#         sleep 2
#     done
#     echo "llama-server on port ${port} did not become ready in time" >&2
#     return 1
# }

# run_llama_container() {
#     local name="$1"
#     local gpu="$2"
#     local port="$3"
#     local model="$4"
#     local agent_name="$5"

#     docker rm -f "${name}" >/dev/null 2>&1 || true

#     local cmd=(
#         docker run -d \
#             --name "${name}" \
#             -p "${port}:${port}" \
#             -v "${MODELS_DIR}:/models" \
#             --gpus "device=${gpu}" \
#             ghcr.nju.edu.cn/ggml-org/llama.cpp:server-cuda \
#             -m "/models/${model}" \
#             -c "${CTX_SIZE}" \
#             --alias "${agent_name}" \
#             --host 0.0.0.0 \
#             --port "${port}" \
#             --n-gpu-layers "${N_GPU_LAYERS}"
#     )
#     echo "Running command:"
#     printf ' %q' "${cmd[@]}"
#     echo

#     "${cmd[@]}"
# }

# PLAN_CID="$(run_llama_container "llama-planner" "${PLAN_GPU}" "${PLAN_PORT}" "${PLAN_MODEL}" "${PLAN_NAME}")"
# docker logs -f llama-planner > "${PLAN_LOG}" 2>&1 &
# echo "planner container=${PLAN_CID}"
# wait_for_llama_server "${PLAN_PORT}"

# CAL_CID="$(run_llama_container "llama-caller" "${CAL_GPU}" "${CAL_PORT}" "${CAL_MODEL}" "${CAL_NAME}")"
# docker logs -f llama-caller > "${CAL_LOG}" 2>&1 &
# echo "caller container=${CAL_CID}"
# wait_for_llama_server "${CAL_PORT}"

# SUM_CID="$(run_llama_container "llama-summarizer" "${SUM_GPU}" "${SUM_PORT}" "${SUM_MODEL}" "${SUM_NAME}")"
# docker logs -f llama-summarizer > "${SUM_LOG}" 2>&1 &
# echo "summarizer container=${SUM_CID}"
# wait_for_llama_server "${SUM_PORT}"

export PYTHONPATH=./

for DOMAIN in in_domain
do 

    cmd=(python inference_utils/toolbench/infer_pipeline_llama.py \
        --planner_model_path "${MODELS_DIR}/${PLAN_MODEL}" \
        --caller_model_path "${MODELS_DIR}/${CAL_MODEL}" \
        --summarizer_model_path "${MODELS_DIR}/${SUM_MODEL}" \
        --data_path "dataset/toolbench/test/${DOMAIN}.json" \
        --assistant_prompt_type "${P_TYPE_PLAN}" \
        --caller_prompt_type "${P_TYPE_CAL}" \
        --conclusion_prompt_type "${P_TYPE_SUM}" \
        --n_threads "${N_THREADS}" \
        --n_batch "${N_BATCH}" \
        --n_ctx "${CTX_SIZE}" \
        --planner_n_gpu_layers "${N_GPU_LAYERS}" \
        --caller_n_gpu_layers "${N_GPU_LAYERS}" \
        --summarizer_n_gpu_layers "${N_GPU_LAYERS}" \
        --max_input_length 3580 \
        --num_infer_samples 2 \
        --output_dir "${LAB_DIR}/${DOMAIN}")
    echo "Running command:"
    printf ' %q' "${cmd[@]}"
    echo

    "${cmd[@]}"
    

    python inference_utils/toolbench/evaluate-multi_agent.py \
        --input_path "${LAB_DIR}/${DOMAIN}/predictions.json" \
        --output_path "${LAB_DIR}/${DOMAIN}/metrics.json"
done
