from src.beam.similarity import TFIDF
from sklearn.datasets import fetch_20newsgroups
from transformers import AutoTokenizer
from beam import beam_logger as logger


if __name__ == '__main__':
    # Load the 20 newsgroups dataset

    logger.info(f"Loaded dataset: newsgroups_train")
    newsgroups_train = fetch_20newsgroups(subset='train')
    newsgroups_test = fetch_20newsgroups(subset='test')

    tokenizer_name = "mistralai/Mistral-7B-v0.1"
    logger.info(f"Loaded tokenizer: {tokenizer_name}")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    logger.info(f"Tokenizing data: newsgroups_train")
    x = tokenizer(newsgroups_train.data)["input_ids"]

    logger.info(f"Tokenizing data: newsgroups_test")
    q = tokenizer(newsgroups_test.data)["input_ids"]
    index = newsgroups_train['target_names']

    # Create a TFIDF model
    tfidf = TFIDF()

    # Fit the model
    vectors = tfidf.fit_transform(x, index)

    # Transform the test data
    scores = tfidf.bm25(q)

    # Print the shape of the transformed data
    print(scores)
