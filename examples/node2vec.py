import networkx as nx
from gensim.models.word2vec import Word2Vec
from pecanpy import pecanpy
import tempfile
from beam import resource


def main():
    # Create the Les Miserables graph
    G = nx.les_miserables_graph()

    # Convert to DataFrame
    df_les_mis = nx.to_pandas_edgelist(G)

    with tempfile.TemporaryDirectory() as tmp_dir:
        file = resource(tmp_dir).joinpath('data.csv')
        file.write(df_les_mis, index=False, header=False, sep='\t')
        g = pecanpy.SparseOTF(p=1, q=1, workers=1, verbose=False)
        g.read_edg(file.str, weighted=True, directed=False)

    walks = g.simulate_walks(num_walks=10, walk_length=80)
    # use random walks to train embeddings
    w2v_model = Word2Vec(walks, vector_size=8, window=3, min_count=0, sg=1, workers=1, epochs=1)

    print(w2v_model)


if __name__ == '__main__':
    main()