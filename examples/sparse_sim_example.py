import torch
from beam import Timer


if __name__ == '__main__':

    M = 40000

    nel = 100
    k1 = 15000
    k2 = 20

    def gen_coo_vectors(k):

        r = []
        c = []
        v = []

        for i in range(k):
            r.append(i * torch.ones(nel, dtype=torch.int64))
            c.append(torch.randint(M, size=(nel,)))
            v.append(torch.randn(nel))

        return torch.sparse_coo_tensor(torch.stack([torch.cat(r), torch.cat(c)]), torch.cat(v), size=(k, M))


    s1 = gen_coo_vectors(k1)
    s2 = gen_coo_vectors(k2)

    # from beam.similarity.sparse import SparseSimilarity as Similarity
    from beam.similarity.sparnn import SparnnSimilarity as Similarity
    sparse_sim = Similarity(metric='cosine', format='coo', vec_size=M, device='cuda')
    sparse_sim.add(s1)

    with Timer(name='sparnn_fit_and_search'):
        sim = sparse_sim.search(s2, k=10)

    s3 = gen_coo_vectors(k2)
    with Timer(name='sparnn__search'):
        sim = sparse_sim.search(s3, k=10)

    print(sim.distance.shape, sim.index.shape)
    print(sim.distance)
    print(sim.index)
    print(sparse_sim.index.shape)
    print(sparse_sim.index)