# Word2Vec
import spacy
import fasttext.util
from statified_Word2vec_de import statified_word2vec
import numpy as np
# Cosine similarity
from scipy import spatial

# Germanet
from germanetpy.germanet import Germanet
from itertools import chain

# Wiki
import wikipediaapi
import pickle

# Sentiment analysis
from germansentiment import SentimentModel

# Common functions
import pandas as pd




class Word2vec:
    """
    A class used to get the relation of hypernym, hyponym, co_hypernym or synonym between DC and EC

    Attributes
    ----------
    df:
        a dataframe of pairs DC and EC
    Methods
    -------
    embedding_fasttext:
        This method applies word2Vec to all the topics with help of fasttext
    embedding_spacy:
        This method applies spacy to get the word embeddings
    embedding_statified:
        This method applies loads the predefined statified contextualized word embeddings
    cos_similarity:
        This method returns the cosine similarity of two vectors. Because that the definition of spatial.distance.cosine
        is 1 - cosine similarity we have to do 1 - spatial.distance.cosine = 1-(1-cosine similarity) = cosine similarity
    """
    def __init__(self, df, model, lang='de'):
        """
        Parameters
        ----------
        df:
            a dataframe of pairs DC and EC
        model:
            a fasttext Word2Vec model
        lang:
            the language for word embedding model
        nlp:
            the nlp library from spacy
        tokenization:
            a dictionary of sentences and their tokens
        lemma:
            a dictionary of words and their lemma
        """
        self.df = df
        self.model = model
        if lang == 'de':
            self.nlp = spacy.load('de_core_news_sm')
        with open("/Users/kangchieh/Downloads/Bachelorarbeit/wiki_concept/de_topic_tokenization.txt", 'rb') as f:
            self.tokenization = pickle.load(f)
        with open("/Users/kangchieh/Downloads/Bachelorarbeit/wiki_concept/expansion_label_de_lemmas.txt", 'rb') as f:
            self.lemma = pickle.load(f)

    def embedding_fasttext(self):
        """
        This method applies fasttext to get the word embeddings

        Returns
        ------
        dict:
            a dictionary contains of every topic and its word embedding
        """
        word2vec = {}
        ft = fasttext.load_model(self.model)  # load model
        for i in range(len(self.df)):
            DC, EC = self.df.at[i, 'DC'], self.df.at[i, 'EC']
            if DC not in word2vec:  # initialize if DC not exists
                word2vec[DC] = ft.get_sentence_vector(DC)
            if EC not in word2vec:  # initialize if EC not exists
                word2vec[EC] = ft.get_sentence_vector(EC)
        return word2vec

    def embedding_spacy(self):
        """
        This method applies spacy to get the word embeddings

        Returns
        ------
        dict:
            a dictionary contains of every topic and its word embedding
        """
        word2vec = {}
        for i in range(len(self.df)):
            DC, EC = self.df.at[i, 'DC'], self.df.at[i, 'EC']
            if DC not in word2vec:  # initialize if DC not exists
                word2vec[DC] = self.nlp(DC).vector
            if EC not in word2vec:  # initialize if EC not exists
                word2vec[EC] = self.nlp(EC).vector
        return word2vec

    def embedding_statified(self):
        """
        This method applies loads the predefined statified contextualized word embeddings

        Returns
        ------
        dict:
            a dictionary contains of every topic and its word embedding
        """
        s = statified_word2vec("/Users/kangchieh/Downloads/Bachelorarbeit/statified word embedding/german_embedding.txt", "/Users/kangchieh/Downloads/Bachelorarbeit/wiki_concept/expansion_label_de_lemmas.txt")
        word2vec = {}
        for i in range(len(self.df)):
            DC, EC = self.df.at[i, 'DC'], self.df.at[i, 'EC']
            if DC not in word2vec:  # initialize if DC not exists
                try:
                    token_lemma_dc = [dc for dc in self.tokenization[DC]]  # get lemma of DC
                    word2vec[DC] = s.fasttext_like_embedding(token_lemma_dc)
                except:  # if some word embedding not exists then return zero vectors
                    word2vec[DC] = np.zeros(768)
            if EC not in word2vec:  # initialize if EC not exists
                try:
                    token_lemma_ec = [ec for ec in self.tokenization[EC]]  # get lemma of EC
                    word2vec[EC] = s.fasttext_like_embedding(token_lemma_ec)
                except:  # if some word embedding not exists then return zero vectors
                    word2vec[EC] = np.zeros(768)
        return word2vec

    def cos_similarity(self, vec1, vec2):
        """
        This method returns the cosine similarity of two vectors. Because that the defintion of spatial.distance.cosine
        is 1 - cosine similarity we have to do 1 - spatial.distance.cosine = 1-(1-cosine similarity) = cosine similarity

        Parameters
        ----------
        vec1: numpy.ndarray
            a numpy array of Word2Vec embedding
        vec2: numpy.ndarray
            a numpy array of Word2Vec embedding

        Returns
        ------
        float:
            the cosine similarity of two vectors
        """
        return 1 - spatial.distance.cosine(vec1, vec2)


class Wordnet_de:
    """
    A class used to get the relation of hypernym, hyponym, co_hypernym or synonym between DC and EC

    Attributes
   ----------
    df:
       a dataframe of pairs DC and EC
    germanet_path:
       the path to load Germatnet model

    Methods
    -------
    get_synset:
       This method returns the synset of DC and EC. Synset : a set of synonyms that share a common meaning
    co_hypernym:
       This method determines if DC a co-hypernym of EC. 1: co_hypernym, 0: not co_hypernym
    hypernym:
       This method determines if DC a hypernym of EC. 1: hypernym, 0: not hypernym
    hyponym:
       This method determines if DC a hyponym of EC. 1: hyponym, 0: not hyponym
    synonym:
       This method determines if DC a synonym of EC. 1: synonym, 0: not synonym
    processing:
       This method processes all the methods
    """
    def __init__(self, df, germanet_path):
        """
        Parameters
        ----------
        df:
            a dataframe of pairs DC and EC
        relation:
            an initialized dataframe initialized for the class
        """
        self.df = df
        self.relation = pd.DataFrame(0, index=df.index, columns=['hypernym', 'hyponym', 'co-hypernym', 'synonym'])
        self.relation = pd.concat([self.df, self.relation], axis=1)
        self.germanet = Germanet(germanet_path)

    def get_synset(self):
        """ This method returns the synset of DC and EC. Synset : a set of synonyms that share a common meaning """
        dc = []
        ec = []
        for i in range(len(self.df)):
            # DC, EC = self.df.at[i, 'DC'].replace(" ", "_"), self.df.at[i, 'EC'].replace(" ", "_")
            DC, EC = self.df.at[i, 'DC'], self.df.at[i, 'EC']
            dc.append(self.germanet.get_synsets_by_orthform(DC))
            ec.append(self.germanet.get_synsets_by_orthform(EC))

        self.df["synset_DC"] = dc
        self.df["synset_EC"] = ec

    def co_hyponym(self):
        """ This method determines if DC a co-hypernym of EC. 1: co_hypernym, 0: not co_hypernym """
        for i in range(len(self.df)):
            DC, EC = self.df.at[i, 'synset_DC'], self.df.at[i, 'synset_EC']
            dc_hyponyms = [syn_dc.all_hyponyms() for syn_dc in DC]
            dc_hyponyms = set(chain.from_iterable(dc_hyponyms))
            ec_hyponyms = [syn_ec.all_hyponyms() for syn_ec in EC]
            ec_hyponyms = set(chain.from_iterable(ec_hyponyms))
            cohyponym = dc_hyponyms.intersection(ec_hyponyms)

            if cohyponym:
                self.relation.at[i, 'co-hypernym'] = 1
            else:
                self.relation.at[i, 'co-hypernym'] = 0

    def hypernym(self):
        """ This method determines if DC a hypernym of EC. 1: hypernym, 0: not hypernym """
        for i in range(len(self.df)):
            DC, EC = self.df.at[i, 'synset_DC'], self.df.at[i, 'synset_EC']
            for syn_ec in EC:
                hyper = syn_ec.all_hypernyms()
                for syn_dc in DC:
                    if syn_dc in hyper:
                        self.relation.at[i, 'hypernym'] = 1
                        break
                else:
                    continue
                break

    def hyponym(self):
        """ This method determines if DC a hyponym of EC. 1: hyponym, 0: not hyponym """
        for i in range(len(self.df)):
            DC, EC = self.df.at[i, 'synset_DC'], self.df.at[i, 'synset_EC']
            for syn_ec in EC:
                hypo = syn_ec.all_hyponyms()
                for syn_dc in DC:
                    if syn_dc in hypo:
                        self.relation.at[i, 'hyponym'] = 1
                        break
                else:
                    continue
                break

    def synonym(self):
        """ This method determines if DC a synonym of EC. 1: synonym, 0: not synonym """
        for i in range(len(self.df)):
            DC, EC = self.df.at[i, 'synset_DC'], self.df.at[i, 'synset_EC']
            dc_synonym = [value for synset in DC for lexunit in synset.lexunits for key, value in
                       lexunit.relations.items()]
            if dc_synonym:
                dc_synonym = set.union(*dc_synonym)
                ec_lexunits = [lexunit for synset in EC for lexunit in synset.lexunits]
                synonym = dc_synonym.intersection(ec_lexunits)
                if synonym:
                    self.relation.at[i, 'synonym'] = 1
                    continue

    def processing(self):
        """
        This method processes all the methods to get hypernym, hyponym, co_hypernym and synonym

        Returns
        ------
        Dataframe:
            a dataframe consists of pairs (DC, EC) and if they are hypernym, hyponym, co_hypernym or synonym
        """
        self.get_synset()
        self.hypernym()
        self.hyponym()
        self.co_hyponym()
        self.synonym()

        return self.relation


class Wiki:
    """
    A class used to get the number of shared categories and shared links of two wiki pages

    Attributes
    ----------
    df:
        a dataframe of pairs DC and EC

    Methods
    -------
    categories:
        This method returns the category lists of a wiki-page
    links:
        This method returns the outlink lists of a wiki-page
    processing:
        This method processes all the methods to get the number of shared categories and shared links
    """
    def __init__(self, df):
        """
        Parameters
        ----------
        df:
            a dataframe of pairs DC and EC
        wiki:
            an initialized dataframe initialized for the class
        """
        # https://wikipedia-api.readthedocs.io/en/latest/README.html
        self.df = df
        self.wiki = pd.DataFrame(0, index=df.index, columns=['shared_categories', 'shared_links'])
        self.wiki = pd.concat([self.df, self.wiki], axis=1)

    def categories(self, page):
        """
        This method returns the category lists of a wiki-page

        Parameters
        ----------
        page: wikipediaapi.WikipediaPage
            a wiki page

        Returns
        -------
        Dict_keys
            the dictionary keys of page's all categorties
        """
        categories = page.categories

        return categories.keys()

    def links(self, page):
        """
        This method returns the outlink lists of a wiki-page

        Parameters
        ----------
        page: wikipediaapi.WikipediaPage
            a wiki page

        Returns
        -------
        Dict_keys
            the dictionary keys of page's all outlinks
        """
        links = page.links

        return links.keys()

    def processing(self, path_cat=None, path_link=None):
        """ This method processes all the methods to get the number of shared categories and shared links """
        wiki = wikipediaapi.Wikipedia('de')
        shared_categories = {}
        shared_links = {}
        if path_cat is None:
            for i in range(len(self.df)):
                DC, EC = self.df.at[i, 'DC'], self.df.at[i, 'EC']
                DC_wiki, EC_wiki = wiki.page(DC), wiki.page(EC)
                # categories
                DC_cat, EC_cat = self.categories(DC_wiki), self.categories(EC_wiki)

                # out links
                DC_outlink, EC_outlink = self.links(DC_wiki), self.links(EC_wiki)

                shared_cat = set(DC_cat).intersection(EC_cat)
                shared_link = set(DC_outlink).intersection(EC_outlink)

                shared_categories[(DC, EC)] = len(shared_cat)
                shared_links[(DC, EC)] = len(shared_link)

            with open("/Users/kangchieh/Downloads/Bachelorarbeit/wiki_concept/wiki/cat_de.pkt",
                          "wb") as f:
                pickle.dump(shared_categories, f)
            with open("/Users/kangchieh/Downloads/Bachelorarbeit/wiki_concept/wiki/link_de.pkt",
                          "wb") as f1:
                pickle.dump(shared_links, f1)

        else:  # load the dictionary
            with open(path_cat, "rb") as f:
                shared_categories = pickle.load(f)
            with open(path_link, "rb") as f1:
                shared_links = pickle.load(f1)
            for i in range(len(self.df)):
                DC, EC = self.df.at[i, 'DC'], self.df.at[i, 'EC']
                if (DC, EC) not in shared_categories:
                    DC_wiki, EC_wiki = wiki.page(DC), wiki.page(EC)
                    # categories
                    DC_cat, EC_cat = self.categories(DC_wiki), self.categories(EC_wiki)

                    # out links
                    DC_outlink, EC_outlink = self.links(DC_wiki), self.links(EC_wiki)

                    shared_cat = set(DC_cat).intersection(EC_cat)
                    shared_link = set(DC_outlink).intersection(EC_outlink)

                    shared_categories[(DC, EC)] = len(shared_cat)
                    shared_links[(DC, EC)] = len(shared_link)

            with open("/Users/kangchieh/Downloads/Bachelorarbeit/wiki_concept/wiki/cat_de.pkt",
                        "wb") as f:
                pickle.dump(shared_categories, f)
            with open("/Users/kangchieh/Downloads/Bachelorarbeit/wiki_concept/wiki/link_de.pkt",
                      "wb") as f1:
                pickle.dump(shared_links, f1)


class Sentiment_Analysis:
    """
    A class used to get the sentiment score of a topic

    Attributes
    ----------
    df:
        a dataframe of pairs DC and EC

    Methods
    -------
    processing:
        This method processes all the methods to get the number of shared categories and shared links
    """
    def __init__(self, df):
        """
        Parameters
        ----------
        df:
            a dataframe of pairs DC and EC
        model:
            the sentiment analysis model
        look_up:
            a look up table for translate sentiment result to numeric values
        """
        self.df = df
        # self.sentiment = pd.DataFrame(0, index=df.index, columns=['DC_sentiment', 'EC_sentiment'])
        # self.sentiment = pd.concat([self.df, self.sentiment], axis=1)
        self.model = SentimentModel()
        self.look_up = {'positive': 1, 'neutral': 0, 'negative': -1}

    def processing(self, path=None):
        """ This method gets the sentiment of all the topics"""
        if path == None:
            topics = list(set(list(self.df.DC.values) + list(self.df.EC.values)))
            topics_sentiment = self.model.predict_sentiment(topics)

            topics_sentiment = [self.look_up[sentiment] for sentiment in topics_sentiment]

            sentiment = dict(zip(topics, topics_sentiment))

            with open("/Users/kangchieh/Downloads/Bachelorarbeit/wiki_concept/sentiment/sentiment_de.pkt",
                      "wb") as f:
                pickle.dump(sentiment, f)
        else:  # load the dictionary
            with open(path, "rb") as f:
                sentiment = pickle.load(f)

            topics = list(set(list(self.df.DC.values) + list(self.df.EC.values)))
            for topic in topics:
                if topic not in sentiment:
                    sentiment[topic] = self.look_up[self.model.predict_sentiment(topic)[0]]
            with open("/Users/kangchieh/Downloads/Bachelorarbeit/wiki_concept/sentiment/sentiment_de.pkt",
                      "wb") as f:
                pickle.dump(sentiment, f)



if __name__ == "__main__":
    df = pd.read_csv("/Users/kangchieh/Downloads/Bachelorarbeit/wiki_concept/filter_de/filter_sim=0.3_freq=0.01.csv", index_col=0)
    df = df.reset_index(drop=True)

    #w = Word2vec(df, '/Users/kangchieh/Downloads/Bachelorarbeit/cc.de.100.bin', 'de')
    w = Wiki(df)
    w.processing()
    # a = w.processing()
    # a.to_csv("/Users/kangchieh/Downloads/Bachelorarbeit/wiki_concept/filter_de/hello.csv")




