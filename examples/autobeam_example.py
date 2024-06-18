
from src.beam.auto import AutoBeam


def main():
    path_to_state = '/home/shared/data/results/enron/models/model_state_subset'
    # path_to_state = '/home/mlspeech/elads/data/enron/models/model_state_subset'
    to_bundle = True
    from_bundle = False
    to_docker = False

    if to_bundle:
        from examples.enron_similarity import EnronTicketSimilarity
        alg = EnronTicketSimilarity.from_path(path_to_state)

        path_to_bundle = f"{path_to_state}_bundle"
        AutoBeam.to_bundle(alg, path_to_bundle)
        print('Bundle saved at:', path_to_bundle)

    if from_bundle:

        print('Loading bundle from:', path_to_state)
        path_to_bundle = f"{path_to_state}_bundle"
        alg = AutoBeam.from_bundle(path_to_bundle)

        res = alg.evaluate(36, known_subset='train', unknown_subset='validation',
                           test_subset='test', k_sparse=10, k_dense=10, threshold=0.5)
        print(res)

    if to_docker:
        from examples.enron_similarity import EnronTicketSimilarity

        # path_to_bundle = f"{path_to_state}_bundle_docker"
        # alg = EnronTicketSimilarity.from_path(path_to_state)
        # AutoBeam.to_docker(alg, bundle_path=path_to_bundle)

        path_to_bundle = f"{path_to_state}_bundle"
        AutoBeam.to_docker(bundle_path=path_to_bundle)
        print('Bundle saved at:', path_to_bundle)


if __name__ == '__main__':
    main()
