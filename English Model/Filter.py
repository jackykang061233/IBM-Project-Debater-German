# regular expression
import re
# os
from os import listdir
from os.path import isfile, join
# NER
import spacy
# word2vec
from HelpFunctions import Word2vec
# stopwords
from nltk.corpus import stopwords
# basic functions
import pickle
import pandas as pd


class Filter:
    def __init__(self, df):
        self.df = df
        self.stop_words = set(stopwords.words('english'))  # english stopwords

    def preprocessing(self, df):
        """ This function finds the stopwords in the sentence"""
        df.loc[:, 'stop_words'] = True
        for i in range(len(df)):
            EC = df.at[i, 'EC']
            if EC.lower() not in self.stop_words:  # if our EC not a stopword
                df.at[i, 'stop_words'] = False

        return df

    def directionality(self, concept, ratio=0.8):
        pass

    def named_entity(self, df):
        """ This function recognizes all named entities"""
        NER = spacy.load("en_core_web_sm")
        df.loc[:, 'ner'] = False
        for i in range(len(df)):
            EC = df.at[i, 'EC']
            if not NER(EC.lower()).ents:  # if not NER exists in EC
                df.at[i, 'ner'] = True

        return df

    def frequency_ratio(self, df, dict_freq=None):
        """ This function calculates the frequency of a word"""
        if dict_freq == None:  # no existing frequency dictionary
            Concept = list(set(list(df.DC.values) + list(df.EC.values)))
            df['DC_freq'] = 0
            df['EC_freq'] = 0
            frequency_counter = {concept: 0 for concept in Concept}

            path = "/Users/kangchieh/Downloads/Bachelorarbeit/wiki_concept/wiki_titles/"
            files = [f for f in listdir(path) if isfile(join(path, f))]  # get all the files in the folder wiki titles

            for concept in Concept:
                counter = 0
                for file in files:
                    if file == ".DS_Store":
                        continue
                    try:
                        text = open(path + file, "r", encoding="utf-8").read()
                        pattern = r"\b" + concept.lower() + r"\b"
                        counter += len(self.re_match(pattern, text.lower())) # accumulate the number of the concepts
                    except FileNotFoundError:
                        pass
                frequency_counter[concept] += counter
        else:  # uses existing frequency dictionary
            with open(dict_freq, "rb") as f:
                frequency_counter = pickle.load(f)

        for i in range(len(df)):
            DC, EC = df.at[i, 'DC'], df.at[i, 'EC']
            df.at[i, 'DC_freq'] = frequency_counter[DC]
            df.at[i, 'EC_freq'] = frequency_counter[EC]

        return df

    def distributional_similarity(self, df):
        """ This function calculates the distributional similarity between DC and EC"""
        df.loc[:, 'distributional_similarity'] = 0.0  # initialization
        # import the class Word2vec from Helfunctions to calculate the word embeddings
        w = Word2vec(df, '/Users/kangchieh/Downloads/Bachelorarbeit/cc.en.100.bin')
        word2vec = w.embedding()
        for i in range(len(df)):
            DC, EC = df.at[i, 'DC'], df.at[i, 'EC']
            df.at[i, 'distributional_similarity'] = w.cos_similarity(word2vec[DC], word2vec[EC])  # calculate cosine similarity

        return df

    def substring(self, df):
        """ EC cannot be the substring of DC or vice versa"""
        df.loc[:, 'substring'] = True
        for i in range(len(df)):
            DC, EC = df.at[i, 'DC'], df.at[i, 'EC']
            if DC.lower() not in EC.lower() and EC.lower() not in DC.lower():  # if EC not a substring of DC and vice versa
                df.at[i, 'substring'] = False

        return df

    def processing(self):
        """ This function processes all the filters at once"""
        preprocess = self.preprocessing(self.df)
        substring = self.substring(preprocess)
        named_entity = self.named_entity(substring)
        dsimilarity = self.distributional_similarity(named_entity)
        frequency = self.frequency_ratio(dsimilarity)

        return frequency

    def filter(self, freq=0.01, dsim=0.3, path="/Users/kangchieh/Downloads/Bachelorarbeit/wiki_concept/filter/frequency.pkl"):
        """ This function filters our given input and return only the ones that fit our criteria"""
        df = pd.read_csv("/Users/kangchieh/Downloads/Bachelorarbeit/wiki_concept/concept_wiki_filter_number.csv")
        with open(path, "rb") as f:
            frequency = pickle.load(f)

        for i in range(len(df)):
            DC, EC = df.at[i, 'DC'], df.at[i, 'EC']
            if not df.at[i, 'stop_words'] and df.at[i, 'ner'] and not df.at[i, 'substring'] \
                    and df.at[i, 'distributional_similarity'] > dsim and min(frequency[DC] / frequency[EC],
                                                                             frequency[EC] / frequency[DC]) > freq:
                df.at[i, 'good expansion'] = 1
            else:
                df.at[i, 'good expansion'] = 0
        return df[df["good expansion"] == 1.0]

    def semantic_relatedness(self, concept):
        pass

    def count_occurrences(self, word, sentence):
        return sentence.lower().count(word)

    def re_match(self, pattern, text):
        """ This functions uses re to find the matched pattern in our text"""
        pattern = re.compile(pattern)
        match = pattern.findall(text)
        return match


if __name__ == "__main__":
    #df = pd.read_csv("/Users/kangchieh/Downloads/Bachelorarbeit/wiki_concept/concept_wiki_con.csv", index_col=0)
    # f = Filter(df)
    # f.frequency_ratio(df)
    # df1 = f.filter()


    # df1.to_csv("/Users/kangchieh/Downloads/Bachelorarbeit/wiki_concept/filter/sim_0.3_freq=0.01.csv")
    path = "/Users/kangchieh/Downloads/Bachelorarbeit/wiki_concept/wiki_titles/"
    files = [f for f in listdir(path) if isfile(join(path, f))]  # get all the files in the folder wiki titles
    words_length = 0

    for file in files:
        if file == ".DS_Store" or ':' in file:
            continue
        try:
            text = open(path + file, "r", encoding="utf-8").read()
            text_length = text.split(' ')
            words_length += len(text_length)
        except FileNotFoundError:
            pass
    print(words_length)

