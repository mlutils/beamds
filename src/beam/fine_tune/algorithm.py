from peft import LoraConfig, get_peft_model

from ..core import Algorithm
# from transformers import Trainer
from transformers import AutoModel, AutoTokenizer, AutoConfig


class FineTuneLLM(Algorithm):

    def __init__(self, hparams, **kwargs):

        config = AutoConfig.from_pretrained(hparams.model)
        model = AutoModel.from_pretrained(hparams.model)
        self._tokenizer = AutoTokenizer.from_pretrained(hparams.model)

        config = LoraConfig(
            r=hparams.lora_r,
            lora_alpha=hparams.lora_alpha,
            target_modules=[
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
            ],
            lora_dropout=hparams.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, config)

        super().__init__(hparams, networks={'llm': model}, **kwargs)


    @property
    def tokenizer(self):
        return self._tokenizer
