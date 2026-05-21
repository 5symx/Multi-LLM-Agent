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

PLAN_MODEL="${PLAN_MODEL:-/data0/ymx/.cache/huggingface/hub/models--iic--alpha-umi-planner-7b}"
CAL_MODEL="${CAL_MODEL:-/data0/ymx/.cache/huggingface/hub/models--iic--alpha-umi-caller-7b}"
SUM_MODEL="${SUM_MODEL:-/data0/ymx/.cache/huggingface/hub/models--iic--alpha-umi-summarizer-7b}"

# PLAN_PORT="${PLAN_PORT:-8001}"
# CAL_PORT="${CAL_PORT:-8002}"
# SUM_PORT="${SUM_PORT:-8003}"
API_HOST="${API_HOST:-www.right.codes/codex}" # codex claude-aws

API_MODEL_NAME="${API_MODEL_NAME:-gpt-5.4-mini}" # claude-haiku-4-5-20251001

LAB_DIR="${LAB_DIR:-output_res/toolbench_api_gpt_100}"
P_TYPE_PLAN="${P_TYPE_PLAN:-toolbench_planner}"
P_TYPE_CAL="${P_TYPE_CAL:-toolbench_caller}"
P_TYPE_SUM="${P_TYPE_SUM:-toolbench_summarizer}"

mkdir -p "${LAB_DIR}"


# echo "Starting summarizer vLLM on port ${SUM_PORT}"
# CUDA_VISIBLE_DEVICES="${SUM_GPU}" vllm serve "${SUM_MODEL}" \
#     --host "${VLLM_HOST}" \
#     --port "${SUM_PORT}" \
#     --gpu-memory-utilization 0.45 \
#     --served-model-name summarizer \
#     --trust-remote-code ${VLLM_EXTRA_ARGS} >"${SUM_LOG}" 2>&1 &
# SUM_PID=$!

export PYTHONPATH=./

# for DOMAIN in in_domain out_of_domain
for DOMAIN in in_domain
do
    EXTRA_API_ARG=()
    if [[ -n "${OPENAI_RIGHT_API_KEY:-}" ]]; then
        EXTRA_API_ARG=(--api_key "${OPENAI_RIGHT_API_KEY}")
    fi

    python inference_utils/toolbench/infer_pipeline_vllm.py \
        --planner_model_name "${API_MODEL_NAME}" \
        --caller_model_name "${API_MODEL_NAME}" \
        --summarizer_model_name "${API_MODEL_NAME}" \
        --planner_tokenizer_name_or_path "${PLAN_MODEL}" \
        --caller_tokenizer_name_or_path "${CAL_MODEL}" \
        --summarizer_tokenizer_name_or_path "${SUM_MODEL}" \
        --planner_base_url "https://${API_HOST}/v1" \
        --caller_base_url "https://${API_HOST}/v1" \
        --summarizer_base_url "https://${API_HOST}/v1" \
        --data_path "dataset/toolbench/test/${DOMAIN}.json" \
        --assistant_prompt_type "${P_TYPE_PLAN}" \
        --caller_prompt_type "${P_TYPE_CAL}" \
        --conclusion_prompt_type "${P_TYPE_SUM}" \
        --max_input_length 3580 \
        --num_infer_samples 100 \
        --output_dir "${LAB_DIR}/${DOMAIN}" \
        "${EXTRA_API_ARG[@]}"
        

    python inference_utils/toolbench/evaluate-multi_agent.py \
        --input_path "${LAB_DIR}/${DOMAIN}/predictions.json" \
        --output_path "${LAB_DIR}/${DOMAIN}/metrics.json"
done
