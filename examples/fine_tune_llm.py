
# from src.beam import as_tensor
# from src.beam.data import BeamData
# from datasets import load_dataset


# hparams = FTLLMHparams(model='codellama/CodeLlama-7b-hf', hf_cache_dir='/mnt/data/models', device=1)
# dataset = FineTuneHFDataset(hparams)
# print(dataset[1])


# dataset = load_dataset("boolq")
# bd = BeamData(data=dataset, quick_getitem=True)
# print(dataset['train'][1, 5, 6])
# bdi = bd[[1, 5, 6]]
#
# print(bdi.values)
# print(bdi)


# dataset = load_dataset("iamtarun/python_code_instructions_18k_alpaca")
# hparams = FTLLMHparams(model='codellama/CodeLlama-7b-hf', hf_cache_dir='/mnt/data/models', device=1, accelerate=True,
#                        amp=False)
#
# alg = FineTuneLLM(hparams)
# sample = alg.tokenizer([dataset['train'][2]['prompt'], dataset['train'][2]['prompt']])
#
# s = as_tensor(sample.data, device=alg.llm.device)
# res = alg.llm(**s)
# alg.llm.save_pretrained('/tmp/models/llm')


from src.beam.fine_tune import FineTuneLLM, FTLLMHparams, FineTuneHFDataset
from src.beam import Experiment


if __name__ == '__main__':
    hparams = FTLLMHparams(model='codellama/CodeLlama-7b-hf', hf_cache_dir='/mnt/data/models', device=1, accelerate=True)
    experiment = Experiment(hparams)
    dataset = FineTuneHFDataset(hparams)
    alg = FineTuneLLM(hparams)
    experiment.fit(alg, dataset)
    print('done!')
