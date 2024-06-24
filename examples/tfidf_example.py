from beam.similarity import TFIDF
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import AutoTokenizer
from beam import beam_logger as logger
from beam.utils import Timer


if __name__ == '__main__':
    # Load the 20 newsgroups dataset

    logger.info(f"Loaded dataset: newsgroups_train")
    newsgroups_train = fetch_20newsgroups(subset='train')
    newsgroups_test = fetch_20newsgroups(subset='test')

    tokenizer_name = "mistralai/Mistral-7B-v0.1"
    logger.info(f"Loaded tokenizer: {tokenizer_name}")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    logger.info(f"Tokenizing data: newsgroups_train")
    x = tokenizer(newsgroups_train.data, add_special_tokens=False)["input_ids"]

    logger.info(f"Tokenizing data: newsgroups_test")
    q = tokenizer(newsgroups_test.data, add_special_tokens=False)["input_ids"]
    index = newsgroups_train['target_names']

    x_str = [' '.join([str(i) for i in xi]) for xi in x]

    with Timer(name='TfidfVectorizer.fit_transform', logger=logger) as t:
        tfidf = TfidfVectorizer()
        vectors = tfidf.fit_transform(x_str)

        logger.info(f"Transformed data: {vectors.shape}")
        logger.critical(f"1x2: {(vectors[0].toarray() * vectors[1].toarray()).sum()}")

    with Timer(name='BeamTFIDF.fit_transform', logger=logger) as t:
        # Create a TFIDF model
        tfidf = TFIDF(sparse_framework='torch', device=0, n_workers=1)

        # Fit the model
        vectors = tfidf.fit_transform(x, index)

        logger.info(f"Transformed data: {vectors.shape}")

        try:
            logger.critical(f"1x2: {(vectors[0].to_dense() * vectors[1].to_dense()).sum()}")
        except AttributeError:
            logger.critical(f"1x2: {(vectors[0].toarray() * vectors[1].toarray()).sum()}")

    with Timer(name='BeamTFIDF.bm25', logger=logger) as t:
        # Transform the test data
        scores = tfidf.bm25(q)

    # Print the shape of the transformed data
    print(scores)
