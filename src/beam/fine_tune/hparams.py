from ..config import UniversalConfig, BeamParam


class FTLLMConfig(UniversalConfig):

    defaults = dict(project_name='fine_tune_llm', training_framework='accelerate', batch_size=2,
                    model_dtype='bfloat16', epoch_length=100,
                    scale_epoch_by_batch_size=False, lr_dense=1e-5, lr_sparse=1e-4, reduction='mean_batch')
    parameters = [BeamParam('model', str, None, 'Model to use for fine-tuning'),
                  BeamParam('prompt_key', str, 'prompt', 'Key to use for the prompt'),
                  BeamParam('completion_key', str, None, 'Key to use for the completion'),
                  BeamParam('lora_alpha', float, 16, 'Lora alpha parameter', tags=['tune']),
                  BeamParam('lora_dropout', float, 0.05, 'Lora dropout', tags=['tune']),
                  BeamParam('lora_r', int, 16, 'Lora r parameter', tags=['tune']),
                  BeamParam('lora_fan_in_fan_out', bool, False, 'Set this to True if the layer to replace stores '
                                                                'weight like (fan_in, fan_out)'),
                  BeamParam('lora_bias', str, 'none', 'Bias type for Lora. Can be ‘none’, '
                                                      '‘all’ or ‘lora_only’. If ‘all’ or ‘lora_only’', tags=['tune']),
                  BeamParam('load_in_8bit', bool, False, 'Load the model in 8bit mode'),
                  BeamParam('modules_to_save', list, None, 'List of modules apart from LoRA layers to be set '
                                                           'as trainable and saved in the final checkpoint'),
                  BeamParam('layers_to_transform', list, None, 'The layer indexes to transform, if this argument '
                                                               'is specified, it will apply the LoRA transformations '
                                                               'on the layer indexes that are specified in this list.'),
                  BeamParam('target_modules', list, None, 'The names of the modules to apply Lora to'),
                  BeamParam('hf_cache_dir', str, None, 'Directory for Huggingface to cache to and load from'),
                  BeamParam('hf_data_dir', str, None, 'Directory for the dataset to load from'),
                  BeamParam('return_overflowing_tokens', bool, False, 'Whether or not to split overflowing tokens into '
                                                                      'their own batch'),
                  BeamParam('context_length', int, 128, 'The maximal context length to train the model with',
                            tags=['tune']),

                  BeamParam('dataset', str, None, 'The dataset which is used for fine-tuning'),
                  ]
