
import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

from ..utils import beam_device
from ..core import Processor
from ..llm import beam_llm, default_tokenizer


class TextSimilarity(Processor):
    def __init__(self, *args, alpha=0.5, dense_model_path="BAAI/bge-base-en-v1.5", tokenizer_path=None,
                 use_dense_model_tokenizer=True, dense_model=None, tokenizer=None,
                 device="cuda",  **kwargs):
        """
        Initialize the RAG (Retrieval-Augmented Generation) retriever.

        Parameters:
        data_train (pd.DataFrame): A dataframe containing the training data with a 'text' column.
        alfa (float): Weighting factor for combining dense and sparse retrieval scores.
        embedding_model (str): The name of the sentence transformer model used for embedding.
        model (str): The name of the transformer model used for causal language modeling.
        device (str): The device to run the models on (e.g., 'cuda:1' for GPU).
        """

        super().__init__(*args, device=device, alpha=alpha, tokenizer_path=tokenizer_path,
                         dense_model_path=dense_model_path, use_dense_model_tokenizer=use_dense_model_tokenizer,
                         **kwargs)

        # Device to run the model
        self.device = beam_device(self.get_hparam('device'))
        # Weighting factor for score combination
        self.alpha = self.get_hparam('alpha')
        # Load the sentence transformer model for embeddings
        if dense_model is None:
            self.dense_model = SentenceTransformer(self.get_hparam('dense_model_path'), device=str(self.device))
        else:
            self.dense_model = dense_model
            self.dense_model.to(self.device)

        if tokenizer is None:
            if self.get_hparam('tokenizer_path') is not None:
                from transformers import PreTrainedTokenizerFast
                self.tokenizer = PreTrainedTokenizerFast(tokenizer_file=self.get_hparam('tokenizer_path'))
            elif self.get_hparam('use_dense_model_tokenizer'):
                self.tokenizer = self.dense_model.tokenize
            else:
                self.tokenizer = default_tokenizer
        else:
            self.tokenizer = tokenizer

        self.bm25 = None
        self.dense_vectors = None
        self.sparse_vectors = None
        self.tokens = []
        self.corpus = []

    def add(self, x):

        self.corpus.extend(x)
        dense_vectors = self.dense_model.encode(x)
        if self.dense_vectors is None:
            self.dense_vectors = dense_vectors
        else:
            self.dense_vectors = np.concatenate((self.dense_vectors, dense_vectors))

        tokens = self.tokenizer(x)['input_ids']
        self.tokens.extend(tokens)
        self.bm25 = BM25Okapi(self.tokens)

    def retrieve_dense(self, question, n_return=10):
        """
        Perform dense retrieval for a given question.

        Parameters:
        question (str): The query question.
        n_return (int): Number of top results to return.

        Returns:
        np.ndarray: Indices of the top retrieved results based on dense retrieval.
        """
        question_embedding = self.dense_model.encode(question)
        self.score_dense = question_embedding @ self.embedding_mat.T
        max_score_dense = np.argsort(self.score_dense)[::-1][:n_return]
        return max_score_dense

    def retrieve_sparse(self, question, n_return=10):
        """
        Perform sparse retrieval for a given question.

        Parameters:
        question (str): The query question.
        n_return (int): Number of top results to return.

        Returns:
        np.ndarray: Indices of the top retrieved results based on sparse retrieval (BM25).
        """
        tokenized_query = question.split(" ")
        self.score_sparse = self.bm25.get_scores(tokenized_query)
        max_scores_sparse = np.argsort(self.score_sparse)[::-1][:n_return]
        return max_scores_sparse

    def retrieve_mix_score(self, question, n_return=10):
        """
        Retrieve documents using a combination of dense and sparse retrieval methods.

        Parameters:
        question (str): The query question.
        n_return (int): Number of top results to return.

        Returns:
        tuple: Indices of the top results based on combined, dense, and sparse scores.
        """
        max_scores_sparse = self.retrieve_sparse(question, n_return=n_return)
        max_score_dense = self.retrieve_dense(question, n_return=n_return)

        self.score_combine = self.alpha * (self.score_dense / self.score_dense.max()) + (1 - self.alpha) * (
                    self.score_sparse / self.score_sparse.max())
        max_scores_combine = np.argsort(self.score_combine)[::-1][:n_return]
        return max_scores_combine, max_score_dense, max_scores_sparse

    def add(self, text, title=[]):
        """
        Add new text data to the training data.

        Parameters:
        text (str): The text to be added.
        title (list): Optional title for the text.
        """
        embed = self.dense_model.encode(text)
        L = self.data_train.shape[0]
        self.data_train.loc[L] = [title, L, text, embed]
        self.embedding_mat = np.concatenate((self.embedding_mat, embed.reshape(1, -1)))
        self.train_bm25()

    def predict(self, question, n_retrieve=10, skip=5):
        pass

    def search(self, question, n_retrieve=10, skip=5):
        return self.predict(question, n_retrieve=n_retrieve, skip=skip)


class RAG(Processor):
    def __init__(self, data_train, alfa=0.5, embedding_model="BAAI/bge-base-en-v1.5",
                 llm="Toten5/Marcoroni-neural-chat-7B-v1", embedding_calc=False, device="cuda:1"):
        """
        Initialize the RAG (Retrieval-Augmented Generation) retriever.

        Parameters:
        data_train (pd.DataFrame): A dataframe containing the training data with a 'text' column.
        alfa (float): Weighting factor for combining dense and sparse retrieval scores.
        embedding_model (str): The name of the sentence transformer model used for embedding.
        model (str): The name of the transformer model used for causal language modeling.
        embedding_calc (bool): Flag to determine if embeddings should be calculated at initialization.
        device (str): The device to run the models on (e.g., 'cuda:1' for GPU).
        """

        # Device to run the model
        self.device = device
        # Dataframe containing training data
        self.data_train = data_train
        # Weighting factor for score combination
        self.alfa = alfa
        # Initialize embedding matrix with zeros
        self.embedding_mat = np.zeros((data_train.shape[0], 768))
        # Load the sentence transformer model for embeddings
        self.embedding_model = SentenceTransformer(embedding_model)
        # Load the transformer model for causal language modeling
        self.llm = beam_llm(llm)

        # Calculate embeddings if required
        if embedding_calc:
            self.calc_embedding()

        # Store embeddings in the embedding matrix
        for i in range(data_train.shape[0]):
            self.embedding_mat[i, :] = self.data_train['embedding'][i]

        # Train the BM25 model for sparse retrieval
        self.train_bm25()

    def train_bm25(self):
        """
        Train the BM25 model using the text data.
        """
        self.corpus = [i for i in self.data_train['text'].values]
        tokenized_corpus = [doc.split(" ") for doc in self.corpus]
        self.bm25 = BM25Okapi(tokenized_corpus)

    def calc_embedding(self):
        """
        Calculate embeddings for each text in the data_train dataframe.
        """
        self.data_train['embedding'] = self.data_train['text'].apply(lambda x: self.embedding_model.encode(x))

    def retrieve_dense(self, question, n_return=10):
        """
        Perform dense retrieval for a given question.

        Parameters:
        question (str): The query question.
        n_return (int): Number of top results to return.

        Returns:
        np.ndarray: Indices of the top retrieved results based on dense retrieval.
        """
        question_embedding = self.embedding_model.encode(question)
        self.score_dense = question_embedding @ self.embedding_mat.T
        max_score_dense = np.argsort(self.score_dense)[::-1][:n_return]
        return max_score_dense

    def retrieve_sparse(self, question, n_return=10):
        """
        Perform sparse retrieval for a given question.

        Parameters:
        question (str): The query question.
        n_return (int): Number of top results to return.

        Returns:
        np.ndarray: Indices of the top retrieved results based on sparse retrieval (BM25).
        """
        tokenized_query = question.split(" ")
        self.score_sparse = self.bm25.get_scores(tokenized_query)
        max_scores_sparse = np.argsort(self.score_sparse)[::-1][:n_return]
        return max_scores_sparse

    def retrieve_mix_score(self, question, n_return=10):
        """
        Retrieve documents using a combination of dense and sparse retrieval methods.

        Parameters:
        question (str): The query question.
        n_return (int): Number of top results to return.

        Returns:
        tuple: Indices of the top results based on combined, dense, and sparse scores.
        """
        max_scores_sparse = self.retrieve_sparse(question, n_return=n_return)
        max_score_dense = self.retrieve_dense(question, n_return=n_return)

        self.score_combine = self.alfa * (self.score_dense / self.score_dense.max()) + (1 - self.alfa) * (
                    self.score_sparse / self.score_sparse.max())
        max_scores_combine = np.argsort(self.score_combine)[::-1][:n_return]
        return max_scores_combine, max_score_dense, max_scores_sparse

    def add(self, text, title=[]):
        """
        Add new text data to the training data.

        Parameters:
        text (str): The text to be added.
        title (list): Optional title for the text.
        """
        embed = self.embedding_model.encode(text)
        L = self.data_train.shape[0]
        self.data_train.loc[L] = [title, L, text, embed]
        self.embedding_mat = np.concatenate((self.embedding_mat, embed.reshape(1, -1)))
        self.train_bm25()

    def answer_llm(self, prompt, max_length=700):
        """
        Generate an answer using the language model.

        Parameters:
        prompt (str): The prompt to be used for generation.
        max_length (int): Maximum length of the generated sequence.

        Returns:
        str: The generated answer.
        """

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        generate_ids = self.llm.generate(inputs.input_ids, max_length=max_length,
                                           pad_token_id=self.tokenizer.eos_token_id)
        new_info_ids = generate_ids[0][len(inputs.input_ids[0]):]
        answer = \
        self.tokenizer.batch_decode([new_info_ids], skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        return answer

    def ask_final_answer(self, question, all_answers, skip=5):
        """
        Generate a final answer based on multiple answers to the same question.

        Parameters:
        question (str): The original question.
        all_answers (list): List of answers generated previously.
        skip (int): Interval for selecting answers from the list.

        Returns:
        str: The final consolidated answer.
        """
        prompt = f"I asked the model the same question based on RAG retrieval text results, the question was: {question}"
        prompt += " and his answers were:"
        for i, ans in enumerate(all_answers[::skip]):
            prompt += f" answer {i}: {ans}."
        prompt += "\nBased on all this information, what should be the answer? Provide only the answer"
        print(f"all_answers: {all_answers[::skip]}")

        final_answer = self.answer_llm(prompt, max_length=5000)
        return final_answer

    def answer_llm_recursive(self, question, unique_order_arr, skip=5):
        """
        Generate an answer recursively based on a sequence of retrieved documents.

        Parameters:
        question (str): The query question.
        unique_order_arr (np.ndarray): Array of indices representing the sequence of documents.

        Returns:
        tuple: The final answer and all intermediate answers.
        """
        answer = ''
        all_answers = []
        for i, data_index in enumerate(unique_order_arr):
            prompt = f"Answer the following question:\n{question}.\nGiven the answer of the model from previous information:\n{answer}\n"
            prompt += f"And given the following new information: {self.data_train.iloc[data_index, 2]}.\nthe answer is:"
            answer = self.answer_llm(prompt)
            print(f"\n\nIteration: {i},  prompt: {prompt}\n\n")
            print(f"Iteration: {i},  Answer: {answer}")
            all_answers.append(answer)
        final_answer = self.ask_final_answer(question, all_answers, skip=skip)
        return final_answer, all_answers

    def answer_question(self, question, n_retrieve=10, skip=5):
        """
        Answer a question using the RAG retrieval method.

        Parameters:
        question (str): The query question.
        n_retrieve (int): Number of documents to retrieve for answering the question.

        Returns:
        tuple: The final answer and all intermediate answers.
        """
        max_scores = self.retrieve_mix_score(question)
        unique_order_arr = np.array(list(dict.fromkeys(np.concatenate(max_scores)).keys()))
        final_answer, all_answers = self.answer_llm_recursive(question, unique_order_arr[:n_retrieve], skip=skip)
        return final_answer, all_answers

