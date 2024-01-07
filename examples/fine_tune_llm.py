
from src.beam.fine_tune import FineTuneLLM, FTLLMConfig, FineTuneHFDataset
from src.beam import Experiment

from src.beam.utils import get_public_ip


def get_paths():

    ip = get_public_ip()

    if ip.startswith('199'):
        hf_cache_dir = '/mnt/data/models/'
    else:
        hf_cache_dir = '/dsi/shared/elads/elads/data/models/'

    return hf_cache_dir

if __name__ == '__main__':

    dataset_name = 'iamtarun/python_code_instructions_18k_alpaca'
    # hparams = FTLLMHparams(model='codellama/CodeLlama-7b-hf', identifier='debug', hf_cache_dir='/mnt/data/models',
    #                        device=0, algorithm=dataset_name.replace('/', '-'), batch_size=6, training_framework='accelerate',
    #                        device_placement=True, dataset=dataset_name, reload=False, resume=-1,
    #                        n_gpus=1)
    hf_cache_dir = get_paths()
    hparams = FTLLMConfig(model='codellama/CodeLlama-7b-hf', identifier='debug', hf_cache_dir=hf_cache_dir,
                          device=0, algorithm=dataset_name.replace('/', '-'), batch_size=16,
                          training_framework='deepspeed', device_placement=True, dataset=dataset_name, reload=False, resume=-1,
                          n_gpus=3, model_dtype='bfloat16')

    experiment = Experiment(hparams)

    # dataset = FineTuneHFDataset(hparams)
    # alg = FineTuneLLM(hparams)
    experiment.fit(alg=FineTuneLLM, dataset=FineTuneHFDataset)
    print('done!')
