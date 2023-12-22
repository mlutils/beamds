from datasets import load_dataset
import torch
from transformers import AutoModelForCausalLM, BitsAndBytesConfig, AutoTokenizer, TrainingArguments
from peft import LoraConfig
from trl import SFTTrainer
from sklearn.model_selection import train_test_split
from datasets import load_dataset
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
import os
os.environ['DISABLE_MLFLOW_INTEGRATION'] = 'TRUE'
os.environ['COMET_MODE'] = 'DISABLED'
os.environ['WANDB_DISABLED'] = 'TRUE'



dataset = load_dataset('flytech/python-codes-25k', split='train')
model_path = '/home/shared/data/models/llm/codellama_7b/'
config = AutoConfig.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path)


def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True)


tokenized_datasets = dataset.map(tokenize_function, batched=True)
train_dataset, test_dataset = train_test_split(tokenized_datasets)

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
)

device_map = {"": 0}

base_model = AutoModelForCausalLM.from_pretrained(
    model_path,
    quantization_config=bnb_config,
    device_map=device_map,
    trust_remote_code=True,
    use_auth_token=True, use_safetensors=False
)



base_model.config.use_cache = False
base_model.config.pretraining_tp = 1

peft_config = LoraConfig(
    lora_alpha=16,
    lora_dropout=0.1,
    r=64,
    bias="none",
    task_type="CAUSAL_LM",
)


tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = 'right'

output_dir = "/tmp/results"

training_args = TrainingArguments(
    output_dir=output_dir,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    logging_steps=10,
    max_steps=50000
)

max_seq_length = 512

trainer = SFTTrainer(
    model=base_model,
    train_dataset=dataset,
    peft_config=peft_config,
    dataset_text_field="text",
    max_seq_length=max_seq_length,
    tokenizer=tokenizer,
    args=training_args,
)

trainer.train()
