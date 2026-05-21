# execution
plan - caller / summerizer / giveup - caller / summerizer
output file: prediction.json
metrics : metrics.json

## performance
reference: ground truth from dataset / in_domain.json
prediction: output from model.

plan_em: plan match
action_em: action name match (for call cases)
easy_f1/hard_f1/f1: action input JSON similarity
rouge: final answer similarity (finish cases)
hallu_rate: predicted action not in tool list.
# reproduce
install env with docker 

## docker env
add dockerfile  / run_docker.sh for cu118 with pytorch 2.0.1

```bash
./run_docker.sh multi-llm-agent:cu118

sh scripts/inference.sh
```
## vllm compatitive
```bash

bash scripts/inference_vllm_docker.sh

docker rm -f $(docker ps -a -q --filter "ancestor=vllm/vllm-openai:latest")

./run_docker.sh multi-llm-agent:cu118

bash scripts/inference_vllm.sh
# vllm_100
```
## llama.cpp compatitive
```bash
bash scripts/inference_llama_docker.sh  -TBTest

# llama_cpp_python
python inference_utils/toolbench/infer_pipeline_llama.py \
  --planner_model_path /path/to/planner.gguf \
  --caller_model_path /path/to/caller.gguf \
  --summarizer_model_path /path/to/summarizer.gguf \
  --data_path dataset/toolbench/test/in_domain.json \
  --assistant_prompt_type toolbench_planner \
  --caller_prompt_type toolbench_caller \
  --conclusion_prompt_type toolbench_summarizer \
  --max_input_length 3580 \
  --num_infer_samples 100 \
  --output_dir output_res/toolbench_llama_local/in_domain

```

## llama cpp python (lora)
```bash
bash scripts/inference_llama.sh
# llama_100 vs 
# llama_lora_100 
```
## opanai - api
change to chat/completion
```bash
bash scripts/inference_llama.sh
# gpt_100
```

# evaluation
explain
plan_em: plan match
action_em: action name match (for call cases)
easy_f1/hard_f1/f1: action input JSON similarity
rouge: final answer similarity (finish cases)
hallu_rate: predicted action not in tool list