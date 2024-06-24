
from beam.fine_tune import FineTuneLLM, FTLLMConfig, FineTuneHFDataset
from beam import Experiment

from beam.utils import get_public_ip


def get_paths():

    ip = get_public_ip()

    if ip.startswith('199'):
        hf_cache_dir = '/mnt/data/models/'
        n_gpus = 3
    else:
        hf_cache_dir = '/dsi/shared/elads/elads/data/models/'
        n_gpus = 4

    return hf_cache_dir, n_gpus


if __name__ == '__main__':

    dataset_name = 'iamtarun/python_code_instructions_18k_alpaca'
    hf_cache_dir, n_gpus = get_paths()

    # hparams = FTLLMConfig(model='codellama/CodeLlama-7b-hf', identifier='debug', hf_cache_dir=hf_cache_dir,
    #                       device=0, algorithm=dataset_name.replace('/', '-'), batch_size=4,
    #                       training_framework='deepspeed', device_placement=True, dataset=dataset_name, reload=False,
    #                       resume=-1, n_gpus=2, model_dtype='float16', distributed_backend='nccl',
    #                       zero_stage=2, context_length=512)

    # hparams = FTLLMConfig(identifier='debug', hf_cache_dir=hf_cache_dir, batch_size=2,
    #                       training_framework='deepspeed', device_placement=True, reload=False,
    #                       resume=-1, device=1, n_gpus=2, model_dtype='float16', distributed_backend='nccl',
    #                       zero_stage=2, model='codellama/CodeLlama-7b-hf', context_length=2048,
    #                       dataset='/home/shared/data/dataset/reverse_engineering/re_v0', algorithm='re_ft',
    #                       prompt_key='decompiled_to_source_prompt',
    #                       completion_key='source')

    hparams = FTLLMConfig(identifier='debug', hf_cache_dir=hf_cache_dir, batch_size=2,
                          training_framework='torch', device_placement=True, reload=False,
                          resume=-1, n_gpus=1, model_dtype='bfloat16', distributed_backend='nccl',
                          zero_stage=2, model='codellama/CodeLlama-7b-hf',
                          dataset='/home/shared/data/dataset/reverse_engineering/re_v0', algorithm='re_ft',
                          prompt_key='decompiled_to_source_prompt')

    experiment = Experiment(hparams)

    # dataset = FineTuneHFDataset(hparams)
    # alg = FineTuneLLM(hparams)
    experiment.fit(alg=FineTuneLLM, dataset=FineTuneHFDataset)
    print('done!')
