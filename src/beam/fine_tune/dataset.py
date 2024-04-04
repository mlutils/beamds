import torch

from ..path import beam_path
from ..dataset import UniversalDataset
from ..data import BeamData
from ..utils import as_tensor
from transformers import AutoTokenizer, AutoConfig
import datasets


class FineTuneHFDataset(UniversalDataset):

    def __init__(self, hparams):

        model_config = AutoConfig.from_pretrained(hparams.model, cache_dir=hparams.hf_cache_dir)
        self.tokenizer = AutoTokenizer.from_pretrained(hparams.model, config=model_config)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        super().__init__(hparams)

        if beam_path(hparams.dataset).is_dir():
            dataset = datasets.load_from_disk(hparams.dataset)
        else:
            dataset = datasets.load_dataset(hparams.dataset, cache_dir=hparams.hf_data_dir)

        self.truncation = False
        self.max_length = None
        if self.hparams.get('context_length') is not None:
            self.max_length = self.hparams.get('context_length')
            self.truncation = True

        self.prompt_key = hparams.get('prompt_key', default='prompt')
        self.completion_key = hparams.get('completion_key', default=None)
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
        prompts = [f"{self.tokenizer.bos_token}{s}{self.tokenizer.eos_token}" for s in sample]
        if self.completion_key is not None:
            prompts = [f"{s}{self.tokenizer.bos_token}{c}{self.tokenizer.eos_token}"
                       for s, c in zip(prompts, sample[self.completion_key])]

        data = self.tokenizer(prompts, padding=True, truncation=self.truncation,
                              max_length=self.max_length, return_tensors='pt',
                              return_overflowing_tokens=self.return_overflowing_tokens).data

        x = data['input_ids'].clone()
        if self.completion_key is not None:
            mask = (x == self.tokenizer.bos_token_id).int()
            first_occurrences = mask.argmax(dim=1)
            contains = mask.any(dim=1)
            result = torch.where(contains, first_occurrences, torch.tensor(x.size(1))).unsqueeze(1)
            x[torch.arange(x.size(1)).unsqueeze(0) <= result] = -100  # -100 is a special token for the loss function
            data['labels'] = x

        return as_tensor(data, device=self.target_device)
