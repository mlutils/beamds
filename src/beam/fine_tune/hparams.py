from ..config import BeamHparams, BeamParam


class FTLLMHparams(BeamHparams):

    defaults = dict(accelerate=True, amp=False, batch_size=4, model_dtype='float16')
    parameters = [BeamParam('model', str, None, 'Model to use for fine-tuning'),
                  BeamParam('lora_alpha', float, 16, 'Lora alpha parameter', tune=True, model=False),
                  BeamParam('lora_dropout', float, 0.05, 'Lora dropout', tune=True, model=False),
                  BeamParam('lora_r', int, 16, 'Lora r parameter', tune=True, model=False),
                  ]
