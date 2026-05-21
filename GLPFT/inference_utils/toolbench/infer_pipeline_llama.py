# Adopted from tatsu-lab@stanford_alpaca. Below is the original copyright:
#    Copyright 2023 Rohan Taori, Ishaan Gulrajani, Tianyi Zhang, Yann Dubois, Xuechen Li
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

"""ToolBench inference pipeline backed by local llama.cpp via llama-cpp-python.

Run three independent role models in-process:
  - planner
  - caller
  - summarizer
"""

from dataclasses import dataclass, field
import copy
import json
import os
import gc
from typing import Optional

from llama_cpp import Llama
from huggingface_hub import hf_hub_download
import transformers
import torch

from utils.prompt_lib import prompt_dict


def _normalize_path(path: str) -> str:
    return os.path.expanduser(path)


@dataclass
class ModelArguments:
    planner_model_name: Optional[str] = field(default="planner")
    caller_model_name: Optional[str] = field(default="caller")
    summarizer_model_name: Optional[str] = field(default="summarizer")

    planner_model_path: str = field(default="models--iic--alpha-umi-planner-7b-Q4_K_M.gguf")
    caller_model_path: str = field(default="models--iic--alpha-umi-caller-7b-Q4_K_M.gguf")
    summarizer_model_path: str = field(default="models--iic--alpha-umi-summarizer-7b-Q4_K_M.gguf")

    planner_lora_path: str = field(default="models--iic--alpha-umi-planner-7b-Q4_K_M.gguf")
    caller_lora_path: str = field(default="models--iic--alpha-umi-caller-7b-Q4_K_M.gguf")
    summarizer_lora_path: str = field(default="models--iic--alpha-umi-summarizer-7b-Q4_K_M.gguf")

    max_tokens: int = field(default=512)
    temperature: float = field(default=0.0)
    top_p: float = field(default=1.0)

    n_threads: int = field(default=2)
    n_batch: int = field(default=512)
    n_ctx: int = field(default=4096)
    n_predict: int = field(default=512)
    planner_n_gpu_layers: int = field(default=-1)
    caller_n_gpu_layers: int = field(default=-1)
    summarizer_n_gpu_layers: int = field(default=-1)


@dataclass
class DataArguments:
    data_path: str = field(
        default=None, metadata={"help": "Path to the training data."}
    )
    lazy_preprocess: bool = False
    max_input_length: int = field(default=1750)
    num_infer_samples: int = field(default=-1)
    planner_prompt_type: str = field(
        default="v1", metadata={"help": "the prompt template"}
    )
    assistant_prompt_type: Optional[str] = field(
        default=None, metadata={"help": "compat alias for planner_prompt_type"}
    )
    caller_prompt_type: str = field(
        default="v1", metadata={"help": "the prompt template"}
    )
    summarizer_prompt_type: str = field(
        default="v1", metadata={"help": "the prompt template"}
    )
    conclusion_prompt_type: Optional[str] = field(
        default=None, metadata={"help": "compat alias for summarizer_prompt_type"}
    )


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(
        default=512,
        metadata={
            "help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )


local_rank = None


def rank0_print(*args):
    if local_rank == 0 or local_rank is None:
        print(*args)


def nested_load_test_data(data_path):
    test_raw_data = []
    if os.path.isdir(data_path):
        for f in os.listdir(data_path):
            temp_test = nested_load_test_data(os.path.join(data_path, f))
            test_raw_data += temp_test
        return test_raw_data
    if os.path.isfile(data_path) and data_path.endswith(".json"):
        rank0_print("Load data from", data_path)
        temp_data = json.load(open(data_path, "r"))
        test_raw_data = temp_data
        return test_raw_data
    return []


def build_infer_samples(data_args):
    print("Loading data...")
    data_paths = data_args.data_path.split(",")
    raw_data = []
    for data_path in data_paths:
        raw_data += nested_load_test_data(data_path=data_path)
    conversations = []
    if data_args.num_infer_samples > 0:
        raw_data = raw_data[: data_args.num_infer_samples]

    prompt_temp = prompt_dict[data_args.planner_prompt_type]
    for d in raw_data:
        c = d["conversations"]
        tool_docs = ""
        for t in d["tools"]:
            tool_docs += json.dumps(t) + "\n"
        tool_names = ", ".join([t["Name"] for t in d["tools"]])
        query_temp = prompt_temp.replace("{doc}", tool_docs).replace(
            "{tool_names}", tool_names
        )
        dispatch = ""
        for j, u in enumerate(c):
            if u["from"] == "assistant":
                if (
                    "Next: caller." in u["value"]
                    or "Next: conclusion." in u["value"]
                    or "Next: give up." in u["value"]
                ):
                    prompt = query_temp.replace("{history}", dispatch)
                    if "Next: caller" in u["value"] or "Next: conclusion." in u["value"]:
                        reference = (
                            u["from"]
                            + ": "
                            + u["value"]
                            + "</s>"
                            + c[j + 1]["from"]
                            + ": "
                            + c[j + 1]["value"]
                            + "</s>"
                        )
                    else:
                        reference = u["value"]
                    conversations.append(
                        {
                            "tools": d["tools"],
                            "instruction": d["instruction"],
                            "history": c[:j],
                            "dispath": copy.deepcopy(dispatch),
                            "model_input": prompt + " assistant: ",
                            "reference": reference,
                        }
                    )
                dispatch += "assistant: " + u["value"] + "</s>"
            elif u["from"] == "user":
                dispatch += "user: " + u["value"] + "</s>"
            elif u["from"] == "observation":
                dispatch += "observation: " + u["value"]
            elif u["from"] == "caller":
                dispatch += "caller: " + u["value"] + "</s>"
            elif u["from"] == "conclusion":
                dispatch += "conclusion: " + u["value"] + "</s>"

    return conversations


def _clean_completion(text: str):
    candidate = text
    if candidate.startswith(": "):
        candidate = candidate[2:]
    if candidate.strip() in ["", ".", ","]:
        candidate = "none"
    return candidate


def _build_prompt(prompt_type, tools, thought, history, role):
    prompt_temp = prompt_dict[prompt_type]
    tool_docs = ""
    for t in tools:
        tool_docs += json.dumps(t) + "\n"
    query = prompt_temp.replace("{doc}", tool_docs)
    tool_names = ", ".join([t["Name"] for t in tools])
    query = query.replace("{tool_names}", tool_names)
    query = query.replace("{thought}", thought)
    query = query.replace("{history}", history)
    return query + f" {role}: "


def _load_llama(model_path: str, lora_path: str, n_threads: int, n_batch: int, n_gpu_layers: int, n_ctx: int, n_predict: int, role: str):
    if not model_path:
        raise ValueError(f"--{role}_model_path is required for infer_pipeline_llama.py")
    # model_path = _normalize_path(model_path)
    # if not os.path.exists(model_path):
    #     raise FileNotFoundError(f"{role} model not found: {model_path}")

    rank0_print(f"Loading {role} model from {model_path}")

    if "/data0/" in model_path or model_path.startswith("/data0"):
        rank0_print("model_path is local")
    else:
        rank0_print("model_path is huggingface")
        model_name_or_path = "Wendy024/iic-alpha-umi-GGUF"
        model_path = hf_hub_download(repo_id=model_name_or_path, filename=model_path)
        
    
    return Llama(
        model_path=model_path,
        lora_path=lora_path,
        n_threads=n_threads,
        n_batch=n_batch,
        n_gpu_layers=n_gpu_layers,
        n_ctx=n_ctx,
        n_predict=n_predict,
    )


def _truncate_prompt(prompt: str, llm: Llama, max_input_length: int) -> str:
    if max_input_length <= 0:
        return prompt
    token_ids = llm.tokenize(prompt.encode("utf-8"), add_bos=True)
    if len(token_ids) <= max_input_length:
        return prompt
    truncated_ids = token_ids[-max_input_length:]
    return llm.detokenize(truncated_ids).decode("utf-8", errors="ignore")


def _llama_completion(llm: Llama, prompt: str, max_tokens: int, temperature: float, top_p: float) -> str:
    response = llm(
        prompt=prompt,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        echo=False,
        stop=None,
    )
    choices = response.get("choices", [])
    if not choices:
        return ""
    return choices[0].get("text", "")


def _release_llama(llm: Optional[Llama], role: str) -> None:
    if llm is None:
        return
    rank0_print(f"Releasing {role} model and clearing caches")
    llm.close()
    del llm
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def infer():
    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments)
    )
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    global local_rank
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))

    if int(os.environ.get("WORLD_SIZE", "1")) > 1 and local_rank != 0:
        return

    if data_args.assistant_prompt_type is not None:
        data_args.planner_prompt_type = data_args.assistant_prompt_type
    if data_args.conclusion_prompt_type is not None:
        data_args.summarizer_prompt_type = data_args.conclusion_prompt_type

    infer_samples = build_infer_samples(data_args)

    process_zero = local_rank == 0 or local_rank is None
    if process_zero and not os.path.exists(training_args.output_dir):
        os.makedirs(training_args.output_dir)

    planner_llm = _load_llama(
        model_args.planner_model_path,
        model_args.planner_lora_path,
        model_args.n_threads,
        model_args.n_batch,
        model_args.planner_n_gpu_layers,
        model_args.n_ctx,
        model_args.n_predict,
        "planner",
    )
    planner_outputs = []
    for sample in infer_samples:
        planner_prompt = _truncate_prompt(
            sample["model_input"], planner_llm, data_args.max_input_length
        )
        text = _llama_completion(
            planner_llm,
            planner_prompt,
            model_args.max_tokens,
            model_args.temperature,
            model_args.top_p,
        )
        planner_outputs.append(text)
    _release_llama(planner_llm, "planner")

    for i, text in enumerate(planner_outputs):
        candidate = _clean_completion(text)
        infer_samples[i]["predictions"] = candidate

    infer_samples_planner = []
    infer_samples_caller = []
    infer_samples_summarizer = []
    for sample in infer_samples:
        if "Next: give up." in sample["predictions"]:
            action_end_idx = sample["predictions"].index("Next: give up.")
            planner_prediction = sample["predictions"][: action_end_idx + len("Next: give up.")]
            sample["predictions"] = planner_prediction
            infer_samples_planner.append(sample)
        elif "Next: conclusion." in sample["predictions"]:
            action_end_idx = sample["predictions"].index("Next: conclusion.")
            planner_prediction = sample["predictions"][: action_end_idx + len("Next: conclusion.")]
            sample["planner_prediction"] = planner_prediction
            query = _build_prompt(
                data_args.summarizer_prompt_type,
                sample["tools"],
                sample["planner_prediction"],
                sample["dispath"] + ("planner: " + sample["planner_prediction"] + "</s>"),
                "conclusion",
            )
            sample["model_input_for_summarizer"] = query
            infer_samples_summarizer.append(sample)
        else:
            if "Next: caller." in sample["predictions"]:
                action_end_idx = sample["predictions"].index("Next: caller.")
                planner_prediction = sample["predictions"][: action_end_idx + len("Next: caller.")]
                sample["planner_prediction"] = planner_prediction
            else:
                planner_prediction = sample["predictions"] + "Next: caller."
                sample["planner_prediction"] = planner_prediction
            query = _build_prompt(
                data_args.caller_prompt_type,
                sample["tools"],
                sample["planner_prediction"],
                sample["dispath"] + ("planner: " + sample["planner_prediction"] + "</s>"),
                "caller",
            )
            sample["model_input_for_caller"] = query
            infer_samples_caller.append(sample)

    if len(infer_samples_caller) != 0:
        caller_llm = _load_llama(
            model_args.caller_model_path,
            model_args.caller_lora_path,
            model_args.n_threads,
            model_args.n_batch,
            model_args.caller_n_gpu_layers,
            model_args.n_ctx,
            model_args.n_predict,
            "caller",
        )
        caller_outputs = []
        for sample in infer_samples_caller:
            caller_prompt = _truncate_prompt(
                sample["model_input_for_caller"], caller_llm, data_args.max_input_length
            )
            text = _llama_completion(
                caller_llm,
                caller_prompt,
                model_args.max_tokens,
                model_args.temperature,
                model_args.top_p,
            )
            caller_outputs.append(text)
        _release_llama(caller_llm, "caller")

        for i, text in enumerate(caller_outputs):
            candidate = _clean_completion(text)
            infer_samples_caller[i]["predictions"] = (
                "asssitant: "
                + infer_samples_caller[i]["planner_prediction"]
                + "</s>caller: "
                + candidate
            )

    if len(infer_samples_summarizer) != 0:
        summarizer_llm = _load_llama(
            model_args.summarizer_model_path,
            model_args.summarizer_lora_path,
            model_args.n_threads,
            model_args.n_batch,
            model_args.summarizer_n_gpu_layers,
            model_args.n_ctx,
            model_args.n_predict,
            "summarizer",
        )
        summarizer_outputs = []
        for sample in infer_samples_summarizer:
            summarizer_prompt = _truncate_prompt(
                sample["model_input_for_summarizer"], summarizer_llm, data_args.max_input_length
            )
            text = _llama_completion(
                summarizer_llm,
                summarizer_prompt,
                model_args.max_tokens,
                model_args.temperature,
                model_args.top_p,
            )
            summarizer_outputs.append(text)
        _release_llama(summarizer_llm, "summarizer")

        for i, text in enumerate(summarizer_outputs):
            candidate = _clean_completion(text)
            infer_samples_summarizer[i]["predictions"] = (
                "asssitant: "
                + infer_samples_summarizer[i]["planner_prediction"]
                + "</s>conclusion: "
                + candidate
            )

    final_infer_sample = infer_samples_caller + infer_samples_planner + infer_samples_summarizer
    if process_zero:
        with open(os.path.join(training_args.output_dir, "predictions.json"), "w") as f:
            json.dump(final_infer_sample, f, indent=4)


if __name__ == "__main__":
    infer()
