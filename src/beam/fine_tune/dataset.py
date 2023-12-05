import torchvision
from ..dataset import UniversalDataset
from ..data import BeamData
from ..utils import DataBatch, as_tensor
from transformers import AutoTokenizer, AutoConfig
import datasets


class FineTuneHFDataset(UniversalDataset):

    def __init__(self, hparams):

        model_config = AutoConfig.from_pretrained(hparams.model, cache_dir=hparams.hf_cache_dir)
        self.tokenizer = AutoTokenizer.from_pretrained(hparams.model, config=model_config)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        super().__init__(hparams, target_device=hparams.device)
        dataset = datasets.load_dataset(hparams.dataset)

        self.truncation = False
        self.max_length = None
        if self.hparams.get('context_length') is not None:
            self.max_length = self.hparams.get('context_length')
            self.truncation = True

        self.return_overflowing_tokens = hparams.get('return_overflowing_tokens', default=False)

        self.data = BeamData({**dataset}, quick_getitem=True)
        if 'test' in self.data.keys():
            test = self.data['test'].index
        else:
            test = hparams.test_size
        if 'validation' in self.data.keys():
            validation = self.data['validation'].index
        else:
            validation = hparams.validation_size

        self.split(validation=validation, test=test, seed=hparams.split_dataset_seed)

    def getitem(self, index):
        sample = self.data[index].data
        # return self.tokenizer(sample['prompt'], padding=True, truncation=True, return_tensors='pt')
        prompts = [f"{s} {self.tokenizer.eos_token}" for s in sample['prompt']]
        data = self.tokenizer(prompts, padding=True, truncation=self.truncation,
                              max_length=self.max_length, return_tensors='pt',
                              return_overflowing_tokens=self.return_overflowing_tokens).data
        return as_tensor(data, device=self.target_device)
