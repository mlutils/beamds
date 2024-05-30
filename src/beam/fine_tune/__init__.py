# Explicit imports for IDE
if len([]):
    from .hparams import FTLLMConfig
    from .algorithm import FineTuneLLM
    from .dataset import FineTuneHFDataset

# look for examples at

# https://hackmd.io/@3tffdwdTRT-Eev0i-1ljZA/SJgQ4dUP2
# https://medium.com/@rajatsharma_33357/fine-tuning-llama-using-lora-fb3f48a557d5
# https://www.philschmid.de/fine-tune-flan-t5-peft
# https://github.com/huggingface/peft/blob/main/examples/lora_dreambooth/train_dreambooth.py

__all__ = ['FTLLMConfig', 'FineTuneLLM', 'FineTuneHFDataset']


def __getattr__(name):
    if name == 'FTLLMConfig':
        from .hparams import FTLLMConfig
        return FTLLMConfig
    elif name == 'FineTuneLLM':
        from .algorithm import FineTuneLLM
        return FineTuneLLM
    elif name == 'FineTuneHFDataset':
        from .dataset import FineTuneHFDataset
        return FineTuneHFDataset
    else:
        raise AttributeError(f"module {__name__} has no attribute {name}")