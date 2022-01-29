# Pickle
import pickle
# Numpy
import numpy as np


class statified_word2vec:
    """
    A class used to get the feature of the german model, other training features like distributional similarity are
    already included in the dataframe before performing this class

    Attributes
    ----------
    embedding_path: String
        A string of the directory path of pretrained word embedding
    lemma_path: String
        A string of the directory path of lemma
    lemma_de: Dictionary
        A dictionary of words and their lemma
    embedding: Dictionary
        A dictionary of words and their word embeddings
    Methods
    -------
    fasttext_like_embedding(words):
        This method calculates the sentence embedding like fasttext
    l2_norm(vec):
        This method calculates the l2 norm of a vector
    max_embedding(words):
        This method takes the maximal value (each dimension) of the word vectors in the sentence
    min_embedding(words):
        This method takes the minimum value (each dimension) of the word vectors in the sentence
    average_embedding(words):
        This method averages the word vectors in the sentence
    """
    def __init__(self, embedding_path, lemma_path):
        """
        Parameters
        ----------
        lemma_de:
            a dictionary of words and their lemma
        embedding:
            a dictionary of words and their embeddings
        """
        self.lemma_de = pickle.load(open(lemma_path, "rb"))
        self.embedding = pickle.load(open(embedding_path, "rb"))

    def fasttext_like_embedding(self, words):
        """
        This method calculates the sentence embedding like fasttext

        Parameters
        ----------
        words: List
            a list of words in a sentence

        Returns
        -------
        Numpy array
            a numpy array of a sentence embedding
        """
        initial_vec = np.zeros(768)
        count = 0  # The number
        for word in words:
            word_embedd = self.embedding[word]
            norm = self.l2_norm(word_embedd)
            if norm > 0:  # if word embedding not equal to 0
                initial_vec = np.add(initial_vec, word_embedd*(1/norm))
                count += 1
        sentence_word_embedd = initial_vec/count
        return sentence_word_embedd

    def l2_norm(self, vec):
        """
        This method calculates the l2 norm of a vector

        Parameters
        ----------
        vec: Numpy array
            a numpy array of word embedding

        Returns
        -------
        Float
            the l2 norm of a vector
        """
        norm = np.linalg.norm(vec)
        return norm

    def max_embedding(self, words):
        """
        This method takes the maximal value (each dimension) of the word vectors in the sentence

        Parameters
        ----------
        words: List
            a list of words in a sentence

        Returns
        -------
        Numpy array
            a numpy array of a sentence embedding
        """
        vec = np.ones(768)*(-np.inf)
        for word in words:
            word_embedd = self.embedding[word]
            if np.any(word_embedd):
                vec = np.maximum(vec, word_embedd)
        return vec

    def min_embedding(self, words):
        """
        This method takes the minimum value (each dimension) of the word vectors in the sentence

        Parameters
        ----------
        words: List
            a list of words in a sentence

        Returns
        -------
        Numpy array
            a numpy array of a sentence embedding
        """
        vec = np.ones(768) * (np.inf)
        for word in words:
            word_embedd = self.embedding[word]
            if np.any(word_embedd):
                vec = np.minimum(vec, word_embedd)
        return vec

    def average_embedding(self, words):
        """
        This method averages the word vectors in the sentence

        Parameters
        ----------
        words: List
            a list of words in a sentence

        Returns
        -------
        Numpy array
            a numpy array of a sentence embedding
        """
        word_embedd = [self.embedding[word] for word in words]
        vec = np.mean(word_embedd, axis=0)
        return vec


if __name__ == "__main__":
    s = statified_word2vec("/Users/kangchieh/Downloads/Bachelorarbeit/statified word embedding/german_embedding.txt", "/Users/kangchieh/Downloads/Bachelorarbeit/wiki_concept/expansion_label_de_lemmas.txt")
    words = ['schlecht', 'Todesstrafe']
    print(s.average_embedding(words))
    # a = np.array([1, 2, 3])
    # print(s.l2_norm(a))



