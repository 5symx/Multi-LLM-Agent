import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from tqdm import tqdm

# Configuration
BASE_MODEL_NAME = "base-model-path"
FT_MODEL_A_NAME = "ft-model-a-path"
FT_MODEL_B_NAME = "ft-model-b-path"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MAX_LENGTH = 512
BATCH_SIZE = 4

# 1. Load Dataset
dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
# Filter out empty lines
texts = [text for text in dataset["text"] if len(text) > 100]

def calculate_kl_divergence(base_model, candidate_model, tokenizer, texts):
    base_model.eval()
    candidate_model.eval()
    total_kl = 0
    count = 0

    with torch.no_grad():
        for i in tqdm(range(0, len(texts), BATCH_SIZE), desc="Calculating KL"):
            batch_texts = texts[i : i + BATCH_SIZE]
            inputs = tokenizer(batch_texts, return_tensors="pt", padding=True, 
                               truncation=True, max_length=MAX_LENGTH).to(DEVICE)
            
            # Get logits
            base_logits = base_model(**inputs).logits
            cand_logits = candidate_model(**inputs).logits

            # Convert logits to probabilities
            # Base model is the target (P), Candidate is the estimate (Q)
            p = F.softmax(base_logits, dim=-1)
            log_q = F.log_softmax(cand_logits, dim=-1)

            # Calculate KL Divergence: sum(P * (logP - logQ))
            # PyTorch's kl_div expects log_target=False by default
            # reduction='batchmean' is mathematically standard
            kl = F.kl_div(log_q, p, reduction='batchmean')
            
            total_kl += kl.item()
            count += 1

    return total_kl / count

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token

# Load Base Model
base_model = AutoModelForCausalLM.from_pretrained(BASE_MODEL_NAME, torch_dtype=torch.float16).to(DEVICE)

# --- Compare Model A ---
model_a = AutoModelForCausalLM.from_pretrained(FT_MODEL_A_NAME, torch_dtype=torch.float16).to(DEVICE)
kl_a = calculate_kl_divergence(base_model, model_a, tokenizer, texts)
print(f"KL Divergence (Base vs Model A): {kl_a:.4f}")

# Clean up memory
del model_a
torch.cuda.empty_cache()

# --- Compare Model B ---
model_b = AutoModelForCausalLM.from_pretrained(FT_MODEL_B_NAME, torch_dtype=torch.float16).to(DEVICE)
kl_b = calculate_kl_divergence(base_model, model_b, tokenizer, texts)
print(f"KL Divergence (Base vs Model B): {kl_b:.4f}")