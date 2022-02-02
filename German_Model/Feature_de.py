# For calculating mean of a list
import statistics

from matplotlib import pyplot as plt
from German_Model.HelpFunctions_de import Wiki, Wordnet_de, Word2vec, Sentiment_Analysis

# Basic functions
import pickle
import numpy as np
import pandas as pd


class GetFeature:
    """
    A class used to get the feature of the german model, other training features like distributional similarity are
    already included in the dataframe before performing this class

    Methods
    -------
    get_wordnet(df):
        This method gets the four features from Germanet
    get_sentiment(df, path):
        This method gets the sentiment of the given topics
    word_embedding(df):
        This method gets the word embedding of the given topics
    process(df, sentiment_path=None):
        This method performs the final process of get the training features
    """

    def __init__(self):
        pass

    # def get_wiki(self, df, path_cat, path_link):
    #     w = Wiki(df)
    #     w.processing(path_cat, path_link)
    #     with open(path_cat, "rb") as f:
    #         cat = pickle.load(f)
    #     with open(path_link, "rb") as f:
    #         link = pickle.load(f)
    #     df['shared_categories'] = df.apply(lambda row: cat.get((row.DC, row.EC)), axis=1)
    #     df['shared_links'] = df.apply(lambda row: link.get((row.DC, row.EC)), axis=1)
    #     return df

    def get_wordnet(self, df):
        """
        This method gets the four features from Germanet

        Parameters
        ----------
        df: Dataframe
            a dataframe with topic pairs and other training features

        Returns
        -------
        Dataframe
            the input dataframe plus the training features from Germanet
        """
        w = Wordnet_de(df)
        df_wordnet = w.processing()
        return df_wordnet

    def get_sentiment(self, df, path):
        """
        This method gets the sentiment of the given topics

        Parameters
        ----------
        df: Dataframe
            a dataframe with topic pairs and other training features
        path: String
            the path of a dictionary with already saved topics and their sentiment

        Returns
        -------
        Dataframe
            the input dataframe plus the sentiment features
        """
        s = Sentiment_Analysis(df)
        s.processing(path)
        with open(path, "rb") as f:
            sentiment = pickle.load(f)
        df['DC_sentiment'] = df["DC"].apply(lambda x: sentiment.get(x))
        df['EC_sentiment'] = df["EC"].apply(lambda x: sentiment.get(x))
        return df

    def word_embedding(self, df):
        """
        This method gets the word embedding of the given topics

        Parameters
        ----------
        df: Dataframe
            a dataframe with topic pairs and other training features

        Returns
        -------
        Dataframe
            the input dataframe plus the word embedding of every topic
        """
        w = Word2vec(df, 'de')
        word2vec = w.embedding_fasttext()  # use fasttext embedding
        dc_embedding = []
        ec_embedding = []
        for i in range(len(df)):
            DC, EC = df.at[i, 'DC'], df.at[i, 'EC']
            dc_embedding.append(list(word2vec[DC]))
            ec_embedding.append(list(word2vec[EC]))

        df['DC_embedding'] = dc_embedding
        df['EC_embedding'] = ec_embedding
        return df

    def processing(self, df, sentiment_path=None):
        """
        This method performs the final processing of get the training features

        Parameters
        ----------
        df: Dataframe
            a dataframe with topic pairs and other training features
        sentiment_path: String
            the path of a dictionary with already saved topics and their sentiment

        Returns
        -------
        Dataframe
            the dataframe plus the word embedding of every topic
        """
        wordnet = self.get_wordnet(df)
        final_result = self.get_sentiment(wordnet, sentiment_path)

        final_result['freq_ratio'] = final_result.apply(
            lambda row: min(row.DC_freq / row.EC_freq,
                            row.EC_freq / row.DC_freq) if row.DC_freq != 0 and row.EC_freq != 0 else 1 / max(
                row.DC_freq, row.EC_freq), axis=1)  # calculating frequency ratio of two words

        return final_result


if __name__ == "__main__":
    pass
