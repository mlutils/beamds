from peft import LoraConfig, get_peft_model

from ..core import Algorithm
from transformers import Trainer, LlamaForCausalLM
from transformers import AutoModel, AutoTokenizer, AutoConfig


class FineTuneLLM(Algorithm):

    def __init__(self, hparams, **kwargs):

        model_config = AutoConfig.from_pretrained(hparams.model)
        model = AutoModel.from_pretrained(hparams.model, config=model_config)
        self._tokenizer = AutoTokenizer.from_pretrained(hparams.model, config=model_config)

        lora_config = LoraConfig(r=hparams.lora_r, lora_alpha=hparams.lora_alpha,
                                 target_modules=hparams.target_modules, lora_dropout=hparams.lora_dropout,
                                 bias=hparams.lora_bias, fan_in_fan_out=hparams.lora_fan_in_fan_out,
                                 modules_to_save=hparams.modules_to_save,
                                 layers_to_transform=hparams.layers_to_transform,
                                 task_type="CAUSAL_LM")
        model = get_peft_model(model, lora_config)

        super().__init__(hparams, networks={'llm': model}, **kwargs)


    @property
    def tokenizer(self):
        return self._tokenizer

    def train_iteration(self, sample=None, label=None, index=None, counter=None, subset=None, training=True, **kwargs):
        net = self.networks['llm']
        res = net(sample, labels=label)
        self.apply(res.loss)

