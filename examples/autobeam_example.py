
from src.beam.auto import AutoBeam


def main():
    # path_to_state = '/home/shared/data/results/enron/models/model_state_subset'
    path_to_state = '/home/mlspeech/elads/data/enron/models/model_state_subset'
    from examples.enron_similarity import EnronTicketSimilarity
    alg = EnronTicketSimilarity.from_path(path_to_state)

    path_to_bundle = f"{path_to_state}_bundle"
    AutoBeam.to_bundle(alg, path_to_bundle)

    print('Bundle saved at:', path_to_bundle)


if __name__ == '__main__':
    main()
