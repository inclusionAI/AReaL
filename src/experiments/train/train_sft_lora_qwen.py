from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, DataCollatorForLanguageModeling
from datasets import load_dataset
from peft import get_peft_model, LoraConfig, TaskType
import torch

# Model checkpoint
MODEL_NAME = "Qwen/Qwen2.5-7B"

# Load tokenizer and model (4-bit optional)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    device_map="auto",
    load_in_4bit=True,  # Set to False if not using QLoRA
    trust_remote_code=True
)

# Apply LoRA config
lora_config = LoraConfig(
    r=64,
    lora_alpha=16,
    target_modules=["c_attn", "q_proj", "v_proj"],  # adjust to Qwen architecture
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.CAUSAL_LM,
)
model = get_peft_model(model, lora_config)

# Load dataset (example: Alpaca format)
dataset = load_dataset("tatsu-lab/alpaca", split="train[:1000]")  # sample subset

# Preprocess
def preprocess(example):
    prompt = example["instruction"] + "\n" + example["input"]
    target = example["output"]
    full_text = prompt + "\n" + target
    return tokenizer(full_text, truncation=True, max_length=1024, padding="max_length")

tokenized_dataset = dataset.map(preprocess, remove_columns=dataset.column_names)

# Data collator
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

# Training args
training_args = TrainingArguments(
    output_dir="./qwen-lora-sft",
    per_device_train_batch_size=2,
    gradient_accumulation_steps=8,
    num_train_epochs=3,
    learning_rate=2e-5,
    fp16=True,  # or bf16=True depending on GPU
    logging_steps=10,
    save_steps=100,
    save_total_limit=2,
    report_to="none",
)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator,
)

trainer.train()
