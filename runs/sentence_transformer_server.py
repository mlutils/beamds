from src.beam import beam_device
from src.beam.config import BeamParam
from src.beam.serve import BeamServeConfig
from src.beam.serve import beam_server


class SentenceTransformerServer(BeamServeConfig):
    parameters = [BeamParam('tokenizer', type=str, default="BAAI/bge-base-en-v1.5", help='Tokenizer model'),
                  BeamParam('st-model-path', type=str, default="BAAI/bge-base-en-v1.5",
                            help='Dense model for text similarity'),
                  BeamParam('st-model-device', type=str, default='cuda', help='Device for dense model'),
                  BeamParam('st-kwargs', type=dict, default={}, help='additional kwargs for the sentence-transformer'),
                  ]


def main():
    config = SentenceTransformerServer()
    device = str(beam_device(config.get('st-model-device')))

    from sentence_transformers import SentenceTransformer
    dense_model = SentenceTransformer(config.get('st-model-path'), device=device, **config.get('st-kwargs', {}))

    beam_server(dense_model, **config)


if __name__ == '__main__':
    main()
