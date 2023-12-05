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

        super().__init__(target_device=hparams.device)
        dataset = datasets.load_dataset(hparams.dataset)

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
        data = self.tokenizer(sample['prompt']).data
        return as_tensor(data, device=self.target_device)
