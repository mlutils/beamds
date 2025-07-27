from beam.dataset import UniversalDataset
from datasets import load_dataset
from beam import as_tensor, beam_device
from transformers import AutoTokenizer
import torch



def get_hf_tokenizer(hparams):
    token = hparams.huggingface_token
    tokenizer = AutoTokenizer.from_pretrained(hparams.tokenizer, token=token, add_eos_token=True,
                           add_bos_token=True,)

    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id

    return tokenizer


class LLMFromScratchDataset(UniversalDataset):
    """
    Dataset class for LLM from scratch experiments.
    This class is used to load and preprocess the dataset for training the LLM.
    """

    def __init__(self, tokenizer, *args, hparams=None, **kwargs):
        super().__init__(*args, hparams=hparams, to_torch=False,
                         target_device=None,
                         **kwargs)
        self._dataset = load_dataset(self.hparams.dataset, split='all')
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self._dataset)

    def getitem(self, idx):
        batch = self._dataset[idx]
        columns = self.hparams.columns
        batch['text'] = ['\n'.join(f"{c}: {si}" for c, si in zip(columns, s)) for s in zip(*[batch[c] for c in columns])]

        c = self.tokenizer(batch['text'], truncation=self.hparams.truncation,
                           padding=self.hparams.padding,
                           max_length=self.hparams.max_length,
                           return_tensors="pt")
        c = {k: c[k] for k in c}
        batch[f"text_pt"] = c

        return batch


