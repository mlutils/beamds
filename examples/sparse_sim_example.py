import torch
from beam.similarity import SparseSimilarity


if __name__ == '__main__':

    M = 40000

    nel = 100
    k1 = 20000
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

    sparse_sim = SparseSimilarity(metric='cosine', format='coo', vec_size=M, device='cuda')
    sparse_sim.add(s1)

    import time
    t = time.time()
    dist, ind = sparse_sim.search(s2, k=10)
    print(time.time() - t)
    print(dist.shape, ind.shape)
    print(dist)
    print(ind)
    print(sparse_sim.index.shape)
    print(sparse_sim.index)