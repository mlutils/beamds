import os
os.environ['PATH'] = f"/usr/local/openmpi/bin:{os.environ['PATH']}"

if 'LD_LIBRARY_PATH' in os.environ:
    os.environ['LD_LIBRARY_PATH'] = f"/usr/local/openmpi/lib:{os.environ['LD_LIBRARY_PATH']}"
else:
    os.environ['LD_LIBRARY_PATH'] = f"/usr/local/openmpi/lib"

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

from src.beam.fine_tune import FineTuneLLM, FTLLMHparams, FineTuneHFDataset
from src.beam import Experiment


if __name__ == '__main__':

    dataset_name = 'iamtarun/python_code_instructions_18k_alpaca'
    # hparams = FTLLMHparams(model='codellama/CodeLlama-7b-hf', identifier='debug', hf_cache_dir='/mnt/data/models',
    #                        device=0, algorithm=dataset_name.replace('/', '-'), batch_size=6, training_framework='accelerate',
    #                        device_placement=True, dataset=dataset_name, reload=False, resume=-1,
    #                        n_gpus=1)
    hparams = FTLLMHparams(model='codellama/CodeLlama-7b-hf', identifier='debug', hf_cache_dir='/mnt/data/models',
                           device=0, algorithm=dataset_name.replace('/', '-'), batch_size=6,
                           training_framework='accelerate', device_placement=True, dataset=dataset_name, reload=False, resume=-1,
                           n_gpus=1, model_dtype='bfloat16')
    experiment = Experiment(hparams)

    # dataset = FineTuneHFDataset(hparams)
    # alg = FineTuneLLM(hparams)
    experiment.fit(alg=FineTuneLLM, dataset=FineTuneHFDataset)
    print('done!')
