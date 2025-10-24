# 🧙 The Elder - Colab Training Cells (Copy & Paste These)

## Instructions:
1. Open your Colab: https://colab.research.google.com/drive/17mXS-su9-9A6E7I42aUXRV1CHsmVDnKD
2. Delete all existing cells
3. Copy each cell below into a new cell in Colab
4. Run in order

---

## CELL 1: Check GPU

```python
import torch
print(f"PyTorch: {torch.__version__}")
print(f"CUDA: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
else:
    print("\n⚠️ NO GPU! Go to: Runtime → Change runtime type → T4 GPU")
    raise SystemExit("GPU required")
```

---

## CELL 2: Load Secrets

```python
import os
from google.colab import userdata

try:
    HF_TOKEN = userdata.get('HF_TOKEN')
    os.environ['HF_TOKEN'] = HF_TOKEN
    os.environ['HUGGING_FACE_HUB_TOKEN'] = HF_TOKEN
    print("✅ HF_TOKEN loaded")
except:
    print("❌ HF_TOKEN not found! Add it in Secrets (🔑 icon)")
    raise

try:
    GH_TOKEN = userdata.get('GH_TOKEN')
    os.environ['GH_TOKEN'] = GH_TOKEN
    print("✅ GH_TOKEN loaded")
except:
    print("⚠️ GH_TOKEN not found (optional)")
    GH_TOKEN = None

GITHUB_USERNAME = "Ishabdullah"
HF_USERNAME = "Ishabdullah"
REPO_NAME = "the-elder-llm"
MODEL_NAME = "The_Elder"
```

---

## CELL 3: Install Packages

```python
# Install with compatible versions for Colab
!pip install -q -U \
    transformers \
    datasets \
    accelerate \
    peft \
    trl \
    bitsandbytes \
    huggingface_hub \
    sentencepiece

print("\n✅ Packages installed!")
print("\n⚠️ IMPORTANT: Click 'RESTART RUNTIME' button above, then continue from CELL 4")
```

**⚠️ AFTER THIS CELL: Click "RESTART RUNTIME" button at the top!**

---

## CELL 4: Verify Installs (Run AFTER restart)

```python
# Re-import after restart
import os
import torch
from google.colab import userdata

# Re-load secrets
HF_TOKEN = userdata.get('HF_TOKEN')
os.environ['HF_TOKEN'] = HF_TOKEN
os.environ['HUGGING_FACE_HUB_TOKEN'] = HF_TOKEN

try:
    GH_TOKEN = userdata.get('GH_TOKEN')
    os.environ['GH_TOKEN'] = GH_TOKEN
except:
    GH_TOKEN = None

GITHUB_USERNAME = "Ishabdullah"
HF_USERNAME = "Ishabdullah"
REPO_NAME = "the-elder-llm"
MODEL_NAME = "The_Elder"

# Verify imports
import transformers
import datasets
import peft
import trl
import bitsandbytes

print("✅ All packages ready!")
print(f"transformers: {transformers.__version__}")
print(f"torch: {torch.__version__}")
print(f"CUDA: {torch.cuda.is_available()}")
```

---

## CELL 5: Clone Repository

```python
!rm -rf the-elder-llm

if GH_TOKEN:
    repo_url = f"https://{GH_TOKEN}@github.com/{GITHUB_USERNAME}/{REPO_NAME}.git"
else:
    repo_url = f"https://github.com/{GITHUB_USERNAME}/{REPO_NAME}.git"

!git clone {repo_url}
%cd the-elder-llm

import json
dataset_path = "data/the_elder_dataset.jsonl"
with open(dataset_path, 'r') as f:
    lines = f.readlines()

print(f"✅ Dataset: {len(lines)} examples")
sample = json.loads(lines[0])
print(f"\nSample Q: {sample['instruction'][:80]}...")
```

---

## CELL 6: Load Model & Tokenizer

```python
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import torch

BASE_MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
print(f"Loading: {BASE_MODEL}")

# 4-bit config
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
)

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, token=HF_TOKEN, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    quantization_config=bnb_config,
    device_map="auto",
    token=HF_TOKEN,
    trust_remote_code=True,
)

model.config.use_cache = False
model.config.pretraining_tp = 1

print(f"✅ Model loaded: {model.get_memory_footprint() / 1e9:.2f} GB")
```

---

## CELL 7: Prepare Dataset

```python
from datasets import load_dataset

dataset = load_dataset('json', data_files='data/the_elder_dataset.jsonl', split='train')

with open('configs/the_elder_system_prompt.txt', 'r') as f:
    system_prompt = f.read().strip()

def format_instruction(sample):
    instruction = sample['instruction']
    input_text = sample.get('input', '')
    output = sample['output']

    user_message = f"{instruction}\n{input_text}" if input_text else instruction

    prompt = f"""<|system|>
{system_prompt}</s>
<|user|>
{user_message}</s>
<|assistant|>
{output}</s>"""

    return {"text": prompt}

formatted_dataset = dataset.map(format_instruction, remove_columns=dataset.column_names)
print(f"✅ Dataset formatted: {len(formatted_dataset)} examples")
```

---

## CELL 8: Configure LoRA

```python
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

model = prepare_model_for_kbit_training(model)

lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()
print("\n✅ LoRA configured")
```

---

## CELL 9: Train! (~30-45 minutes)

```python
from transformers import TrainingArguments
from trl import SFTTrainer

training_args = TrainingArguments(
    output_dir="./the-elder-output",
    num_train_epochs=3,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    lr_scheduler_type="cosine",
    warmup_ratio=0.03,
    weight_decay=0.01,
    logging_steps=10,
    save_strategy="steps",
    save_steps=100,
    save_total_limit=3,
    fp16=True,
    optim="paged_adamw_32bit",
    max_grad_norm=0.3,
    group_by_length=True,
    report_to="none",
)

trainer = SFTTrainer(
    model=model,
    train_dataset=formatted_dataset,
    tokenizer=tokenizer,
    args=training_args,
    dataset_text_field="text",
    max_seq_length=512,
    packing=False,
)

print("="*80)
print("🚀 TRAINING THE ELDER")
print("="*80)
print(f"Dataset: {len(formatted_dataset)} examples")
print(f"Epochs: 3")
print(f"Effective batch size: 16")
print("="*80)

trainer.train()

print("\n✅ TRAINING COMPLETE!")
```

---

## CELL 10: Save & Merge

```python
from peft import PeftModel

print("Saving LoRA...")
trainer.model.save_pretrained("./the-elder-lora")
tokenizer.save_pretrained("./the-elder-lora")

print("Merging...")
base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    torch_dtype=torch.float16,
    device_map="auto",
    token=HF_TOKEN,
    trust_remote_code=True,
)

merged_model = PeftModel.from_pretrained(base_model, "./the-elder-lora")
merged_model = merged_model.merge_and_unload()

merged_model.save_pretrained("./the-elder-merged", safe_serialization=True)
tokenizer.save_pretrained("./the-elder-merged")

print("✅ Model merged and saved")
```

---

## CELL 11: Test The Elder

```python
from transformers import pipeline

generator = pipeline(
    "text-generation",
    model=merged_model,
    tokenizer=tokenizer,
    max_new_tokens=150,
    temperature=0.7,
    do_sample=True,
    top_p=0.95,
)

test_questions = [
    "What is true strength?",
    "How should I respond when someone insults me?",
]

for q in test_questions:
    prompt = f"<|system|>\n{system_prompt}</s>\n<|user|>\n{q}</s>\n<|assistant|>\n"
    response = generator(prompt)[0]['generated_text']
    answer = response.split("<|assistant|>")[-1].split("</s>")[0].strip()
    print(f"Q: {q}")
    print(f"A: {answer}\n")
    print("-"*80 + "\n")
```

---

## CELL 12: Push to Hugging Face

```python
from huggingface_hub import HfApi, create_repo, login

login(token=HF_TOKEN)

repo_id = f"{HF_USERNAME}/{MODEL_NAME}"
create_repo(repo_id, token=HF_TOKEN, private=False, exist_ok=True)

print(f"Pushing to {repo_id}...")
merged_model.push_to_hub(repo_id, token=HF_TOKEN, use_auth_token=True)
tokenizer.push_to_hub(repo_id, token=HF_TOKEN, use_auth_token=True)

print(f"\n✅ Model live at: https://huggingface.co/{repo_id}")
```

---

## CELL 13: Create Model Card

```python
model_card = f"""---
license: apache-2.0
language: [en]
tags: [philosophy, wisdom, stoicism, bushido, native-american-wisdom]
base_model: {BASE_MODEL}
---

# 🧙 The Elder - Wisdom Guide LLM

A philosophical AI guide trained on Bushido, Stoicism, and Native American wisdom.

## Usage

```python
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

model = AutoModelForCausalLM.from_pretrained("{repo_id}")
tokenizer = AutoTokenizer.from_pretrained("{repo_id}")
generator = pipeline("text-generation", model=model, tokenizer=tokenizer)

response = generator("What is true strength?", max_new_tokens=150)
print(response[0]['generated_text'])
```

## Mobile (GGUF)

Download `The_Elder.gguf` for Android LLM apps (SmolChat, LM Studio, etc.)

## Training

- Base: TinyLlama-1.1B-Chat
- Method: LoRA (r=16, α=32)
- Dataset: 50+ wisdom Q&A pairs
- Epochs: 3

---

*"The warrior trains not to avoid falling, but to rise each time with greater wisdom."*
"""

with open("./the-elder-merged/README.md", "w") as f:
    f.write(model_card)

api = HfApi()
api.upload_file(
    path_or_fileobj="./the-elder-merged/README.md",
    path_in_repo="README.md",
    repo_id=repo_id,
    token=HF_TOKEN,
)

print("✅ Model card uploaded")
```

---

## CELL 14: Convert to GGUF (~10 minutes)

```python
print("Installing llama.cpp...")
!git clone https://github.com/ggerganov/llama.cpp 2>/dev/null || true
!cd llama.cpp && make 2>/dev/null || true
!pip install -q gguf

print("\nConverting to FP16 GGUF...")
!python llama.cpp/convert.py ./the-elder-merged --outtype f16 --outfile ./the-elder-f16.gguf

print("\nQuantizing to Q4_K_M...")
!./llama.cpp/quantize ./the-elder-f16.gguf ./The_Elder.gguf Q4_K_M

import os
gguf_size = os.path.getsize("./The_Elder.gguf") / (1024 * 1024)
print(f"\n✅ GGUF created: {gguf_size:.2f} MB")
```

---

## CELL 15: Upload GGUF

```python
print("Uploading GGUF...")
api = HfApi()
api.upload_file(
    path_or_fileobj="./The_Elder.gguf",
    path_in_repo="The_Elder.gguf",
    repo_id=repo_id,
    token=HF_TOKEN,
)

print(f"\n✅ GGUF uploaded!")
print(f"\n📥 Download: https://huggingface.co/{repo_id}/resolve/main/The_Elder.gguf")
```

---

## CELL 16: Final Summary

```python
print("="*80)
print("🎉 THE ELDER - TRAINING COMPLETE")
print("="*80)
print(f"\n✅ Model: https://huggingface.co/{repo_id}")
print(f"✅ GGUF: https://huggingface.co/{repo_id}/resolve/main/The_Elder.gguf")
print(f"✅ Size: {gguf_size:.2f} MB")
print("\n📱 Install on Android:")
print("  1. Download The_Elder.gguf")
print("  2. Install SmolChat or LM Studio")
print("  3. Load GGUF file")
print("  4. Chat with The Elder!")
print("\n✨ May The Elder guide you on your path ✨")
print("="*80)
```

---

## 🎯 REMEMBER:

1. **Before running**: Enable GPU (Runtime → Change runtime type → T4 GPU)
2. **Add secrets**: Click 🔑 icon, add `HF_TOKEN` and `GH_TOKEN`
3. **After Cell 3**: Click "RESTART RUNTIME" button
4. **After restart**: Continue from Cell 4
5. **Total time**: ~1 hour

---

## ✅ That's all 16 cells!

Copy them one by one into your Colab notebook and run in order.
