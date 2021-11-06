# Word2Vec
import fasttext.util
# cosine similarity
import wikipediaapi
from scipy import spatial
# wordnet
from pygermanet import load_germanet
# basic functions
import re
import pandas as pd
pd.set_option('display.max_columns', 10)  # show at most 10 columns


class Word2vec:
    """
    A class used to get the relation of hypernym, hyponym, co_hypernym or synonym between DC and EC

    Attributes
    ----------
    df:
        a dataframe of pairs DC and EC
    model:
        a fasttext Word2Vec model
    Methods
    -------
    clean_sentence:
        This method cleans the sentence: every word should be lower case and has no space.
        No special symbol allowed
    preprocessing:
        This method gets all the sentences
    embedding:
        This method applies word2Vec to all the concepts
    cos_similarity:
        This method returns the cosine similarity of two vectors. Because that the definition of spatial.distance.cosine
        is 1 - cosine similarity we have to do 1 - spatial.distance.cosine = 1-(1-cosine similarity) = cosine similarity
    """
    def __init__(self, df, model):
        """
        Parameters
        ----------
        df:
            a dataframe of pairs DC and EC
        model:
            a fasttext Word2Vec model
        """
        self.df = df
        self.model = model

    def clean_sentence(self, sentence):
        """
        This method cleans the sentence: every word should be lower case and has no space.
        No special symbol allowed

        Parameters
        ----------
        sentence: String
            the string of a sentence

        Returns
        ------
        String:
            the string of a sentence that has already been adjusted with the above condictions
        """
        sentence = sentence.lower().strip()
        sentence = re.sub(r'[^a-z0-9\s]', ' ', sentence)
        return sentence.split()

    def preprocessing(self, sentences):
        """ This method gets all the sentences """
        split_sentences = []
        for sentence in sentences:
            split_sentences.append(self.clean_sentence(sentence))
        return split_sentences

    def embedding(self):
        """
        This method applies word2Vec to all the concepts

        Returns
        ------
        dict:
            a dictionary contains of every concept and its Word2Vec embedding
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


class Wordnet:
    """
    A class used to get the relation of hypernym, hyponym, co_hypernym or synonym between DC and EC

    Attributes
   ----------
    df:
       a dataframe of pairs DC and EC

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
    def __init__(self, df):
        """
        Parameters
        ----------
        df:
            a dataframe of pairs DC and EC
        relation:
            a dataframe initialized with pairs (DC, EC) which later will be filled with 0s and 1s
            basen on their relations on hypernym, hyponym, co_hypernym or synonym
        gn:
            Germatnet model
        """
        self.df = df
        self.relation = pd.DataFrame(0, index=df.index, columns=['hypernym', 'hyponym', 'co-hypernym', 'synonym'])
        self.relation = pd.concat([self.df, self.relation], axis=1)
        self.gn = load_germanet()

    def get_synset(self):
        """ This method returns the synset of DC and EC. Synset : a set of synonyms that share a common meaning """
        dc = []
        ec = []
        for i in range(len(self.df)):
            DC, EC = self.df.at[i, 'DC'].lower().replace(" ", "_"), self.df.at[i, 'EC'].lower().replace(" ", "_")
            dc.append(self.gn.synsets(DC))
            ec.append(self.gn.synsets(EC))
        self.df["synset_DC"] = dc
        self.df["synset_EC"] = ec

    def co_hypernym(self):
        """ This method determines if DC a co-hypernym of EC. 1: co_hypernym, 0: not co_hypernym """
        for i in range(len(self.df)):
            DC, EC = self.df.at[i, 'synset_DC'], self.df.at[i, 'synset_EC']
            cohyper = list(set(DC).intersection(EC))
            if cohyper:
                self.relation.at[i, 'co-hypernym'] = 1
            else:
                self.relation.at[i, 'co-hypernym'] = 0

    def hypernym(self):
        """ This method determines if DC a hypernym of EC. 1: hypernym, 0: not hypernym """
        for i in range(len(self.df)):
            DC, EC = self.df.at[i, 'synset_DC'], self.df.at[i, 'synset_EC']
            for syn_ec in EC:
                hyper = syn_ec.hypernyms()
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
                hypo = syn_ec.hyponyms()
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
            for syn_ec in EC:
                synonym = syn_ec.lemmas()
                for syn_dc in DC:
                    if syn_dc in synonym:
                        self.relation.at[i, 'synonym'] = 1
                        break
                else:
                    continue
                break

    def processing(self):
        """
        This method processes all the methods

        Returns
        ------
        Dataframe:
            a dataframe consists of pairs (DC, EC) and if they are hypernym, hyponym, co_hypernym or synonym
        """
        self.get_synset()
        self.hypernym()
        self.hyponym()
        self.co_hypernym()
        self.synonym()

        return self.relation


class Wiki:
    def __init__(self, df):
        # https://wikipedia-api.readthedocs.io/en/latest/README.html
        self.df = df
        self.wiki = pd.DataFrame(0, index=df.index, columns=['shared_categories', 'shared_links'])
        self.wiki = pd.concat([self.df, self.wiki], axis=1)

    def categories(self, page):
        """ This method returns the category lists of a wiki-page"""
        categories = page.categories

        return categories.keys()

    def links(self, page):
        """ This method returns the links lists of a wiki-page"""
        links = page.links

        return links.keys()

    def processing(self):
        """ This method process all the above methods return the number of shared values"""
        wiki = wikipediaapi.Wikipedia('en')
        for i in range(len(self.df)):
            DC, EC = wiki.page(self.df.at[i, 'DC']), wiki.page(self.df.at[i, 'EC'])
            # categories
            DC_cat, EC_cat = self.categories(DC), self.categories(EC)

            # out links
            DC_outlink, EC_outlink = self.links(DC), self.links(EC)

            shared_cat = set(DC_cat).intersection(EC_cat)
            shared_link = set(DC_outlink).intersection(EC_outlink)

            self.wiki.at[i, 'shared_categories'] = len(shared_cat)
            self.wiki.at[i, 'shared_links'] = len(shared_link)

        return self.wiki


if __name__ == "__main__":
    df = pd.read_csv("/Users/kangchieh/Downloads/Bachelorarbeit/wiki_concept/concept_wiki_filter_number.csv", index_col=0)
    # w = Wordnet(df)
    # df = w.processing()
    wik = Wiki(df)



