from beam import NeuralAlgorithm
from examples.llm_from_scratch.model import get_hf_model
from examples.llm_from_scratch.dataset import get_hf_tokenizer


# class HFLLMFromScratch(NeuralAlgorithm):
#
#     def __init__(self, hparams):
#
#         tokenizer = get_hf_tokenizer(hparams)
#         model = get_hf_model(tokenizer, hparams)
#
#         super().__init__(hparams, networks=model)
#
#
#     def
#