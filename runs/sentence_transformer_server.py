from beam import beam_device
from beam.config import BeamParam
from beam.serve import BeamServeConfig
from beam.serve import beam_server
from beam import logger


import beam

logger.info(f"beam version: {beam.__version__}")


class SentenceTransformerServer(BeamServeConfig):
    parameters = [
                  BeamParam('tokenizer', type=str, default="BAAI/bge-base-en-v1.5", help='Tokenizer model'),
                  BeamParam('st-model-path', type=str, default="BAAI/bge-base-en-v1.5",
                            help='Dense model for text similarity'),
                  BeamParam('st-model-device', type=str, default='cuda', help='Device for dense model'),
                  BeamParam('st-kwargs', type=dict, default={}, help='additional kwargs for the sentence-transformer'),
                  ]


def main():
    config = SentenceTransformerServer()
    logger.info(config)

    device = str(beam_device(config.get('st-model-device')))

    from sentence_transformers import SentenceTransformer
    dense_model = SentenceTransformer(config.get('st-model-path'), device=device, **config.get('st-kwargs', {}))

    beam_server(dense_model, **config)


if __name__ == '__main__':
    main()
