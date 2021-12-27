# Word2Vec
import pickle

import numpy as np

from Statified_Word2vec import Statified_Word2vec
import fasttext.util
# cosine similarity
import spacy
import wikipediaapi
from scipy import spatial
# wordnet
from nltk.corpus import wordnet as wn
# basic functions
import re
import pandas as pd
pd.set_option('display.max_columns', 10)


class Word2vec:
    def __init__(self, df):
        self.df = df
        self.model = '/Users/kangchieh/Downloads/Bachelorarbeit/cc.en.100.bin'
        self.nlp = spacy.load('en_core_web_sm')

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

    def embedding_fasttext(self):
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

    def embedding_spacy(self):
        """ This function applies word2Vec to all the concepts"""
        word2vec = {}
        for i in range(len(self.df)):
            DC, EC = self.df.at[i, 'DC'], self.df.at[i, 'EC']
            if DC not in word2vec:  # initialize if DC not exists
                word2vec[DC] = self.nlp(DC).vector
            if EC not in word2vec:  # initialize if EC not exists
                word2vec[EC] = self.nlp(EC).vector
        return word2vec

    def embedding_statified(self):
        word2vec = {}
        s = Statified_Word2vec()
        for i in range(len(self.df)):
            DC, EC = self.df.at[i, 'DC'], self.df.at[i, 'EC']
            if DC not in word2vec:  # initialize if DC not exists
                try:
                    word2vec[DC] = s.word2vec(DC)
                except KeyError:
                    word2vec[DC] = np.zeros(768)
            if EC not in word2vec:  # initialize if EC not exists
                try:
                    word2vec[EC] = s.word2vec(EC)
                except KeyError:
                    word2vec[EC] = np.zeros(768)
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

    def co_hyponym(self):
        """ This function determines if DC a co-hypernym of EC"""
        for i in range(len(self.df)):
            DC, EC = self.df.at[i, 'synset_DC'], self.df.at[i, 'synset_EC']
            cohyponym = [syn_dc.lowest_common_hypernyms(syn_ec) for syn_dc in DC for syn_ec in EC]
            if cohyponym:
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
        self.co_hyponym()
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

    def processing(self, path_cat=None, path_link=None):
        """ This functions process all the above functions return the number of shared values"""
        wiki = wikipediaapi.Wikipedia('en')
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

            with open("/Users/kangchieh/Downloads/Bachelorarbeit/wiki_concept/wiki/cat_en.pkt",
                      "wb") as f:
                pickle.dump(shared_categories, f)
            with open("/Users/kangchieh/Downloads/Bachelorarbeit/wiki_concept/wiki/link_en.pkt",
                      "wb") as f1:
                pickle.dump(shared_links, f1)

        else:
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

            with open("/Users/kangchieh/Downloads/Bachelorarbeit/wiki_concept/wiki/cat_en.pkt",
                      "wb") as f:
                pickle.dump(shared_categories, f)
            with open("/Users/kangchieh/Downloads/Bachelorarbeit/wiki_concept/wiki/link_en.pkt",
                      "wb") as f1:
                pickle.dump(shared_links, f1)


if __name__ == "__main__":
    df = pd.read_csv("/Users/kangchieh/Downloads/Bachelorarbeit/wiki_concept/concept_wiki_filter_number.csv", index_col=0)
    # w = Wordnet(df)
    # df = w.processing()
    # wiki = Wiki(df)
    sync_dc = wn.synsets('democracy')
    sync_ec = wn.synsets('gun')
    print(sync_dc)
    print(sync_ec)

    hypernym_dc = [syn.hypernyms() for syn in sync_dc]
    hypernym_ec = [syn.hypernyms() for syn in sync_ec]
    print(hypernym_dc)
    print(hypernym_ec)

    common_hypernym = [dc.lowest_common_hypernyms(ec) for dc in sync_dc for ec in sync_ec]
    print(common_hypernym)




