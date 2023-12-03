from peft import LoraConfig, get_peft_model

from ..core import Algorithm
from ..path import local_copy, beam_path
from transformers import Trainer, LlamaForCausalLM
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
import transformers


class FineTuneLLM(Algorithm):

    def __init__(self, hparams, **kwargs):

        model_config = AutoConfig.from_pretrained(hparams.model, cache_dir=hparams.hf_cache_dir)
        # architecture = getattr(transformers, model_config.architectures[0])

        model = AutoModelForCausalLM.from_pretrained(hparams.model, config=model_config,
                                                     load_in_8bit=hparams.load_in_8bit)
        self._tokenizer = AutoTokenizer.from_pretrained(hparams.model, config=model_config)

        lora_config = LoraConfig(r=hparams.lora_r, lora_alpha=hparams.lora_alpha,
                                 target_modules=hparams.target_modules, lora_dropout=hparams.lora_dropout,
                                 bias=hparams.lora_bias, fan_in_fan_out=hparams.lora_fan_in_fan_out,
                                 modules_to_save=hparams.modules_to_save,
                                 layers_to_transform=hparams.layers_to_transform,
                                 task_type="CAUSAL_LM")

        model = get_peft_model(model, lora_config)
        super().__init__(hparams, networks={'llm': model}, **kwargs)

        # # self.llm = get_peft_model(model, lora_config)
        # super().__init__(hparams, **kwargs)


    @property
    def tokenizer(self):
        return self._tokenizer

    def train_iteration(self, sample=None, label=None, index=None, counter=None, subset=None, training=True, **kwargs):
        net = self.networks['llm']
        res = net(sample, labels=label)
        self.apply(res.loss)

    def save_checkpoint(self, path=None, networks=True, optimizers=True, schedulers=True,
                        processors=True, scaler=True, scalers=True, swa_schedulers=True, swa_networks=True,
                        hparams=True, aux=None, pickle_model=False):

        tmp_path = beam_path('io:///')
        with local_copy(tmp_path, as_beam_path=False) as local_path:
            self.networks['llm'].save_pretrained(local_path)

        if networks:
            if aux is None:
                aux = {}
            aux['lora'] = tmp_path.data

        super().save_checkpoint(path=path, networks=False, optimizers=optimizers, schedulers=schedulers,
                        processors=processors, scaler=scaler, scalers=scalers, swa_schedulers=swa_schedulers, swa_networks=swa_networks,
                        hparams=hparams, aux=aux, pickle_model=pickle_model)

        raise NotImplementedError

    def load_checkpoint(self, path=None, networks=True, optimizers=True, schedulers=True,
                        processors=True, scaler=True, scalers=True, swa_schedulers=True, swa_networks=True,
                        hparams=True, aux=None, pickle_model=False):

        aux = super().load_checkpoint(path=path, networks=False, optimizers=optimizers, schedulers=schedulers,
                        processors=processors, scaler=scaler, scalers=scalers, swa_schedulers=swa_schedulers, swa_networks=swa_networks,
                        hparams=hparams, aux=True, pickle_model=pickle_model)

        tmp_path = beam_path('io:///', data=aux['lora'])
        with local_copy(tmp_path, as_beam_path=False) as local_path:
            self.networks['llm'].load_pretrained(local_path)
