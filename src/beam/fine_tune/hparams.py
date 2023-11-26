from ..config import BeamHparams, BeamParam


class FTLLMHparams(BeamHparams):

    defaults = dict(accelerate=True, amp=False, batch_size=4, model_dtype='float16')
    parameters = [BeamParam('model', str, None, 'Model to use for fine-tuning'),
                  BeamParam('lora_alpha', float, 16, 'Lora alpha parameter', tune=True, model=False),
                  BeamParam('lora_dropout', float, 0.05, 'Lora dropout', tune=True, model=False),
                  BeamParam('lora_r', int, 16, 'Lora r parameter', tune=True, model=False),
                  BeamParam('lora_fan_in_fan_out', bool, False, 'Set this to True if the layer to replace stores '
                                                                'weight like (fan_in, fan_out)'),
                  BeamParam('lora_bias', str, 'none', 'Bias type for Lora. Can be ‘none’, '
                                                      '‘all’ or ‘lora_only’. If ‘all’ or ‘lora_only’', tune=True),
                  BeamParam('modules_to_save', list, None, 'List of modules apart from LoRA layers to be set '
                                                           'as trainable and saved in the final checkpoint'),
                  BeamParam('layers_to_transform', list, None, 'The layer indexes to transform, if this argument '
                                                               'is specified, it will apply the LoRA transformations '
                                                               'on the layer indexes that are specified in this list.'),
                  BeamParam('target_modules', list, None, 'The names of the modules to apply Lora to'),

                  ]
