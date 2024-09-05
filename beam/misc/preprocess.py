from ..type import BeamType, Types


def svd_preprocess(x):

    x_type = BeamType.check_minor(x)

    if x_type.minor == Types.tensor:
        crow_indices = x.crow_indices().numpy()
        col_indices = x.col_indices().numpy()
        values = x.values().numpy()

        # Create a SciPy CSR matrix
        from scipy.sparse import csr_matrix
        x = csr_matrix((values, col_indices, crow_indices), shape=x.size())
    return x