
from src.beam.fine_tune import FineTuneLLM, FTLLMHparams, FineTuneHFDataset
from src.beam import Experiment


if __name__ == '__main__':

    dataset_name = 'iamtarun/python_code_instructions_18k_alpaca'
    hparams = FTLLMHparams(model='codellama/CodeLlama-7b-hf', identifier='debug', hf_cache_dir='/mnt/data/models',
                           device=1, algorithm=dataset_name.replace('/', '-'),
                           accelerate=True, dataset=dataset_name, reload=False, resume=0)
    experiment = Experiment(hparams)

    # dataset = FineTuneHFDataset(hparams)
    # alg = FineTuneLLM(hparams)
    experiment.fit(alg=FineTuneLLM, dataset=FineTuneHFDataset)
    print('done!')
