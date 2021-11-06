# Word2Vec
import fasttext.util
# cosine similarity
import wikipediaapi
from scipy import spatial
# wordnet
from nltk.corpus import wordnet as wn
# basic functions
import re
import pandas as pd
pd.set_option('display.max_columns', 10)


class Word2vec:
    def __init__(self, df, model):
        self.df = df
        self.model = model

    def clean_sentence(self, sentence):
        """
        This function cleans the sentence: every word should be lower case and has no space.
        No special symbol allowed
        """
        sentence = sentence.lower().strip()
        sentence = re.sub(r'[^a-z0-9\s]', ' ', sentence)
        return sentence.split()

    def preprocessing(self, sentences):
        """ This function gets all the sentences"""
        split_sentences = []
        for sentence in sentences:
            split_sentences.append(self.clean_sentence(sentence))
        return split_sentences

    def embedding(self):
        """ This function applies word2Vec to all the concepts"""
        word2vec = {}
        ft = fasttext.load_model(self.model)
        for i in range(len(self.df)):
            DC, EC = self.df.at[i, 'DC'], self.df.at[i, 'EC']
            if DC not in word2vec:  # initialize if DC not exists
                word2vec[DC] = ft.get_sentence_vector(DC)
            if EC not in word2vec:  # initialize if EC not exists
                word2vec[EC] = ft.get_sentence_vector(EC)
        return word2vec

    def cos_similarity(self, a, b):
        """
        This function returns the cosine similarity of two vectors. Because that the defintion of spatial.distance.cosine
        is 1-cosine similarity we have to do 1-spatial.distance.cosine = 1-(1-cosine similarity) = cosine similarity
        """
        return 1 - spatial.distance.cosine(a, b)


class Wordnet:
    def __init__(self, df):
        self.df = df
        self.relation = pd.DataFrame(0, index=df.index, columns=['hypernym', 'hyponym', 'co-hypernym', 'synonym'])
        self.relation = pd.concat([self.df, self.relation], axis=1)

    def get_synset(self):
        """ This function returns the synset of DC and EC. Synset : a set of synonyms that share a common meaning"""
        dc = []
        ec = []
        for i in range(len(self.df)):
            DC, EC = self.df.at[i, 'DC'].lower().replace(" ", "_"), self.df.at[i, 'EC'].lower().replace(" ", "_")
            dc.append(wn.synsets(DC))
            ec.append(wn.synsets(EC))
        self.df["synset_DC"] = dc
        self.df["synset_EC"] = ec

    def co_hypernym(self):
        """ This function determines if DC a co-hypernym of EC"""
        for i in range(len(self.df)):
            DC, EC = self.df.at[i, 'synset_DC'], self.df.at[i, 'synset_EC']
            cohyper = list(set(DC).intersection(EC))
            if cohyper:
                self.relation.at[i, 'co-hypernym'] = 1
            else:
                self.relation.at[i, 'co-hypernym'] = 0

    def hypernym(self):
        """ This function determines if DC a hypernym of EC"""
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
        """ This function determines if DC a hyponym of EC"""
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
        """ This function determines if DC a synonym of EC"""
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
        """ This function processes all the functions"""
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
        """ This functions returns the category lists of a wiki-page"""
        categories = page.categories

        return categories.keys()

    def links(self, page):
        """ This functions returns the links lists of a wiki-page"""
        links = page.links

        return links.keys()

    def processing(self):
        """ This functions process all the above functions return the number of shared values"""
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



