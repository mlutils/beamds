from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.processors import TemplateProcessing

from ..logging import beam_logger as logger
from ..processor import Processor
from ..type import check_type, Types



class BeamTokenizer(Processor):
    def __init__(self, *args, vocab_size=30000, unk_token='<unk>', bos_token='<s>', eos_token='</s>',
                 mask_token='<mask>', pad_token='<pad>', max_token_length=None, min_token_length=None,
                 min_frequency=2, special_tokens=None, bpe_kwargs=None, **kwargs):
        super().__init__(*args, vocab_size=vocab_size, unk_token=unk_token, bos_token=bos_token,
                         eos_token=eos_token, mask_token=mask_token, pad_token=pad_token, min_frequency=min_frequency,
                         special_tokens=special_tokens, bpe_kwargs=bpe_kwargs, max_token_length=max_token_length ,
                         min_token_length=min_token_length, **kwargs)

        self.special_tokens = self.get_hparam('special_tokens', special_tokens)
        if self.special_tokens is None:
            self.special_tokens = []

        self._vocab_size = self.hparams.vocab_size
        self.unk_token = self.hparams.unk_token
        self.bos_token = self.hparams.bos_token
        self.eos_token = self.hparams.eos_token
        self.mask_token = self.hparams.mask_token
        self.pad_token = self.hparams.pad_token
        self.min_frequency = self.hparams.min_frequency
        self.max_token_length  = self.hparams.max_token_length
        self.min_token_length = self.hparams.min_token_length

        self.special_tokens = ([self.unk_token, self.bos_token, self.eos_token, self.mask_token, self.pad_token]
                               + self.special_tokens)

        bpe_kwargs = self.get_hparam('bpe_kwargs', {})

        bpe = BPE(unk_token=self.unk_token, **bpe_kwargs)
        self._tokenizer = Tokenizer(bpe)
        self._filtered_tokens = None

        # Add special tokens
        self._tokenizer.add_special_tokens(self.special_tokens)

        self._trainer = BpeTrainer(special_tokens=self.special_tokens, vocab_size=self._vocab_size,
                                   min_frequency=self.min_frequency, max_token_length=self.max_token_length )

        self._tokenizer.pre_tokenizer = Whitespace()
        self._tokenizer.post_processor = TemplateProcessing(single=f"{self.bos_token} $A {self.eos_token}",
                                                            special_tokens=[(self.bos_token, 1),
                                                                            (self.eos_token, 2)])

    def encode(self, x):
        tokens = self._tokenizer.encode(x).ids
        if self._filtered_tokens is not None:
            tokens = [token for token in tokens if token in self._filtered_tokens]
        return tokens

    def _filter_tokens_by_length(self):
        # Filter vocabulary based on token length constraints (N1 and N2)
        vocab = self._tokenizer.get_vocab()
        if self.min_token_length is not None or self.max_token_length  is not None:
            filtered_vocab = {token: idx for token, idx in vocab.items()
                              if (self.min_token_length is None or len(token) >= self.min_token_length) and
                              (self.max_token_length  is None or len(token) <= self.max_token_length )}
            # Update the tokenizer's vocabulary
            self._filtered_tokens = set(filtered_vocab.values())
            logger.info(f"Filtered vocabulary size: {len(filtered_vocab)}")

    def train(self, texts):
        self._tokenizer.train_from_iterator(texts, self._trainer)
        # Filter tokens by length after training
        self._filter_tokens_by_length()
        return self

    def __call__(self, x):
        x_type = check_type(x)
        if x_type.major == Types.array:
            return [self.encode(item) for item in x]
        return self.encode(x)

    @property
    def vocab(self):
        vocab = self._tokenizer.get_vocab()
        if self._filtered_tokens is not None:
            vocab = {k: v for k, v in vocab.items() if v in self._filtered_tokens}
        return vocab

    @property
    def vocab_size(self):
        return len(self.vocab)

    @property
    def inverse_vocab(self):
        return {v: k for k, v in self.vocab.items()}

