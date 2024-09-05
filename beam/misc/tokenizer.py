from beam.base import BeamBase

from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace


class BeamTokenizer(BeamBase):
    def __init__(self, *args, vocab_size=30000, unk_token='<unk>', min_frequency=2, special_tokens=None, **kwargs):
        super().__init__(*args, **kwargs)

        self.special_tokens = self.get_hparam('special_tokens', special_tokens)
        if self.special_tokens is None:
            self.special_tokens = ['<s>', '<pad>', '</s>', '<unk>', '<mask>']
        self.vocab_size = self.get_hparam('vocab_size', vocab_size)
        self.unk_token = self.get_hparam('unk_token', unk_token)
        self.min_frequency = self.get_hparam('min_frequency', min_frequency)
        self._tokenizer = Tokenizer(BPE(unk_token=self.unk_token))
        self._trainer = BpeTrainer(special_tokens=self.special_tokens, vocab_size=self.vocab_size,
                                   min_frequency=self.min_frequency)
        self._tokenizer.pre_tokenizer = Whitespace()

    def train(self, texts):
        self._tokenizer.train_from_iterator(texts, self._trainer)
        return self

    def __call__(self, x):
        return self._tokenizer.encode(x).ids
