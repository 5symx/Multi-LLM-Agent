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

"""ToolBench inference pipeline backed by vLLM OpenAI-compatible HTTP servers.

Run three independent vLLM servers, one per role:
  - planner
  - caller
  - summarizer

Each server can live on a separate port and can expose the OpenAI completions
API at ``/v1/completions``.
"""

from dataclasses import dataclass, field
import copy
import gc
import json
import os
import time
from typing import Any, Dict, Optional

import requests
import torch
import transformers
from rouge import Rouge
from torch.utils.data import Dataset
from transformers.trainer_pt_utils import LabelSmoother

from utils.prompt_lib import prompt_dict


def evaluate_rougel(cand_list: list, ref_list: list):
    if len(ref_list) == 0:
        return 0
    rouge = Rouge()
    rouge_score = rouge.get_scores(hyps=cand_list, refs=ref_list, avg=True)
    rougel = rouge_score["rouge-l"]["f"]
    return rougel


IGNORE_TOKEN_ID = LabelSmoother.ignore_index


@dataclass
class ModelArguments:
    planner_model_name: Optional[str] = field(default="planner")
    caller_model_name: Optional[str] = field(default="caller")
    summarizer_model_name: Optional[str] = field(default="summarizer")
    planner_tokenizer_name_or_path: Optional[str] = field(default=None)
    caller_tokenizer_name_or_path: Optional[str] = field(default=None)
    summarizer_tokenizer_name_or_path: Optional[str] = field(default=None)

    planner_base_url: str = field(default="http://127.0.0.1:8001/v1")
    caller_base_url: str = field(default="http://127.0.0.1:8002/v1")
    summarizer_base_url: str = field(default="http://127.0.0.1:8003/v1")

    api_key: str = field(default="")
    max_tokens: int = field(default=512)# 1024
    temperature: float = field(default=0.0)
    top_p: float = field(default=1.0)
    request_timeout: float = field(default=600.0)
    max_retries: int = field(default=3)
    retry_sleep: float = field(default=2.0)


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


class InferDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, raw_data, tokenizer: transformers.PreTrainedTokenizer, args):
        super(InferDataset, self).__init__()
        self.tokenizer = tokenizer
        self.raw_data = raw_data
        self.args = args

    def __len__(self):
        return len(self.raw_data)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        return self.raw_data[i]


class Collator(object):
    def __init__(self, tokenizer, args):
        self.tokenizer = tokenizer
        self.args = args

    def __call__(self, features):
        input_ids = [self.tokenizer.encode(x) for x in features]
        max_len = max([len(t) for t in input_ids])
        max_len = min(self.args.max_input_length, max_len)
        new_input_ids = []
        for t in input_ids:
            if len(t) > max_len:
                new_t = t[-max_len:]
            else:
                new_t = [self.tokenizer.pad_token_id] * (max_len - len(t)) + t
            new_input_ids.append(new_t)
        input_ids = torch.LongTensor(new_input_ids)
        attention_mask = torch.ne(input_ids, self.tokenizer.pad_token_id)
        attention_mask = torch.zeros_like(input_ids).masked_fill(attention_mask, 1)
        return dict(input_ids=input_ids, attention_mask=attention_mask)


def _openai_headers(api_key: str):
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    return headers


def _post_openai_completion(
    base_url: str,
    model: str,
    prompt: str,
    api_key: str,
    max_tokens: int,
    temperature: float,
    top_p: float,
    request_timeout: float,
    max_retries: int,
    retry_sleep: float,
):
    url = base_url.rstrip("/") + "/chat/completions"

    rank0_print(f"\n[DEBUG] send to following url:  {url}")
    payload = {
        "model": model,
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": top_p,
        "n": 1,
        "stream": False,
    }

    last_error = None
    for attempt in range(max_retries):
        try:
            print(f"[DEBUG] request Headers: {_openai_headers(api_key)}")
            print(f"[DEBUG] request Payload: {json.dumps(payload, indent=2, ensure_ascii=False)}")
            
            print(f"[DEBUG] sending POST request (Attempt {attempt + 1}/{max_retries})...")
            response = requests.post(
                url,
                headers=_openai_headers(api_key),
                json=payload,
                timeout=request_timeout,
            )
            response.raise_for_status()
            data = response.json()
            choices = data.get("choices", [])
            if not choices:
                raise RuntimeError(f"No choices returned from {url}: {data}")
            # return choices[0].get("text", "")
            return choices[0].get("message", {}).get("content", "")
        except Exception as exc:
            last_error = exc
            if attempt + 1 < max_retries:
                time.sleep(retry_sleep)
            else:
                raise RuntimeError(
                    f"vLLM request failed after {max_retries} attempts for {url}"
                ) from last_error


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


def _load_tokenizer(tokenizer_name_or_path: Optional[str], model_name: str, role: str):
    source = tokenizer_name_or_path or model_name
    try:
        return transformers.AutoTokenizer.from_pretrained(source, use_fast=False)
    except Exception as exc:
        raise RuntimeError(
            f"Failed to load tokenizer for {role} from '{source}'. "
            f"Set --{role}_tokenizer_name_or_path to the served model path."
        ) from exc


def _truncate_prompt(prompt: str, tokenizer, max_input_length: int) -> str:
    if max_input_length <= 0:
        return prompt
    input_ids = tokenizer.encode(prompt)
    if len(input_ids) <= max_input_length:
        return prompt
    truncated_ids = input_ids[-max_input_length:]
    return tokenizer.decode(
        truncated_ids,
        skip_special_tokens=False,
        clean_up_tokenization_spaces=False,
    )


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

    rank0_print("load vLLM endpoints begin")
    planner_tokenizer = _load_tokenizer(
        model_args.planner_tokenizer_name_or_path,
        model_args.planner_model_name,
        "planner",
    )
    caller_tokenizer = _load_tokenizer(
        model_args.caller_tokenizer_name_or_path,
        model_args.caller_model_name,
        "caller",
    )
    summarizer_tokenizer = _load_tokenizer(
        model_args.summarizer_tokenizer_name_or_path,
        model_args.summarizer_model_name,
        "summarizer",
    )

    process_zero = local_rank == 0 or local_rank is None
    if process_zero and not os.path.exists(training_args.output_dir):
        os.makedirs(training_args.output_dir)

    planner_outputs = []
    for sample in infer_samples:
        planner_prompt = _truncate_prompt(
            sample["model_input"], planner_tokenizer, data_args.max_input_length
        )
        text = _post_openai_completion(
            base_url=model_args.planner_base_url,
            model=model_args.planner_model_name,
            prompt=planner_prompt,
            api_key=model_args.api_key,
            max_tokens=model_args.max_tokens,
            temperature=model_args.temperature,
            top_p=model_args.top_p,
            request_timeout=model_args.request_timeout,
            max_retries=model_args.max_retries,
            retry_sleep=model_args.retry_sleep,
        )
        planner_outputs.append(text)

    for i, text in enumerate(planner_outputs):
        candidate = _clean_completion(text)
        infer_samples[i]["predictions"] = candidate

    rank0_print("finish planner")

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
        caller_outputs = []
        for sample in infer_samples_caller:
            caller_prompt = _truncate_prompt(
                sample["model_input_for_caller"], caller_tokenizer, data_args.max_input_length
            )
            text = _post_openai_completion(
                base_url=model_args.caller_base_url,
                model=model_args.caller_model_name,
                prompt=caller_prompt,
                api_key=model_args.api_key,
                max_tokens=model_args.max_tokens,
                temperature=model_args.temperature,
                top_p=model_args.top_p,
                request_timeout=model_args.request_timeout,
                max_retries=model_args.max_retries,
                retry_sleep=model_args.retry_sleep,
            )
            caller_outputs.append(text)

        for i, text in enumerate(caller_outputs):
            candidate = _clean_completion(text)
            infer_samples_caller[i]["predictions"] = (
                "asssitant: "
                + infer_samples_caller[i]["planner_prediction"]
                + "</s>caller: "
                + candidate
            )

    if len(infer_samples_summarizer) != 0:
        summarizer_outputs = []
        for sample in infer_samples_summarizer:
            summarizer_prompt = _truncate_prompt(
                sample["model_input_for_summarizer"], summarizer_tokenizer, data_args.max_input_length
            )
            text = _post_openai_completion(
                base_url=model_args.summarizer_base_url,
                model=model_args.summarizer_model_name,
                prompt=summarizer_prompt,
                api_key=model_args.api_key,
                max_tokens=model_args.max_tokens,
                temperature=model_args.temperature,
                top_p=model_args.top_p,
                request_timeout=model_args.request_timeout,
                max_retries=model_args.max_retries,
                retry_sleep=model_args.retry_sleep,
            )
            summarizer_outputs.append(text)

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
