from beam.config import NNExperimentConfig, BeamParam


class LLMFromScratchConfig(NNExperimentConfig):
    parameters = [
        BeamParam('dataset', type=str, default='Open-Orca/OpenOrca',
                  help='Dataset to use for training the LLM'),
        BeamParam('tokenizer', type=str, default='meta-llama/Llama-3.2-3B',
                  help='Tokenizer to use for training the LLM'),

        BeamParam('max_length', type=int, default=1024,
                  help='Maximum length of the input sequences'),
        BeamParam('padding', type=str, default='longest',
                    help='Padding strategy to use for the input sequences'),
        BeamParam('truncation', type=bool, default=True,
                    help='Whether to truncate the input sequences to the maximum length'),

        BeamParam('columns', type=list, default=['system_prompt', 'question', 'response'],
                  help='Columns to use from the dataset for training the LLM'),

        BeamParam('huggingface_token', type=str, default=None,
                    help='Hugging Face token for accessing private models and datasets'),

        BeamParam('model_name', type=str, default='meta-llama/Llama-3.2-3B',
                    help='Name of the model to use for training the LLM'),

        BeamParam('dtype', type=str, default='bfloat16',
                    help='Data type to use for training the LLM, e.g., bfloat16, float16, float32'),

    ]