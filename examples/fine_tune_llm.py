from src.beam.fine_tune import FineTuneLLM, FTLLMHparams
from src.beam import as_tensor

from datasets import load_dataset
dataset = load_dataset("iamtarun/python_code_instructions_18k_alpaca")
hparams = FTLLMHparams(model='codellama/CodeLlama-7b-hf', hf_cache_dir='/mnt/data/models', device=1, accelerate=True,
                       amp=False)

alg = FineTuneLLM(hparams)
sample = alg.tokenizer([dataset['train'][2]['prompt'], dataset['train'][2]['prompt']])

s = as_tensor(sample.data, device=alg.llm.device)
res = alg.llm(**s)
alg.llm.save_pretrained('/tmp/models/llm')
