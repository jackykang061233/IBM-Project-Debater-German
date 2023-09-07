# regular expression
import re
# os
import string
from os import listdir
from os.path import isfile, join
# NER
import spacy
# word2vec
from English_Model.HelpFunctions import Word2vec
# stopwords
from nltk.corpus import stopwords
# basic functions
import pickle
import pandas as pd


class Filter_en:
    def __init__(self):
        self.stop_words = set(stopwords.words('english'))  # english stopwords

    def preprocess(self, df):
        df = df[df['EC'].notna()]
        df = df.reset_index(drop=True)
        # for i in range(len(df)):
        #     df.at[i, 'EC'] = df.at[i, 'EC'].translate(str.maketrans('', '', string.punctuation))
        #     df.at[i, 'DC'], df.at[i, 'EC'] = ' '.join(df.at[i, 'DC'].split()), ' '.join(df.at[i, 'EC'].split())
        return df

    def stop_word(self, df):
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
        Concept = list(set(list(df.DC.values) + list(df.EC.values)))
        path = "wiki_concept/wiki_titles/"
        files = [f for f in listdir(path) if isfile(join(path, f))]  # get all the files in the folder wiki titles
        if dict_freq == None:  # no existing frequency dictionary
            df['DC_freq'] = 0
            df['EC_freq'] = 0
            frequency_counter = {concept: 0 for concept in Concept}

            for concept in Concept:
                print(concept)
                counter = 0
                for file in files:
                    if file == ".DS_Store":
                        continue
                    try:
                        text = open(path + file, "r", encoding="utf-8").read()
                        pattern = r"\b" + concept.lower() + r"\b"
                        counter += len(self.re_match(pattern, text.lower()))  # accumulate the number of the concepts
                    except FileNotFoundError:
                        pass
                frequency_counter[concept] += counter

        else:  # uses existing frequency dictionary
            with open(dict_freq, "rb") as f:
                frequency_counter = pickle.load(f)
            for concept in Concept:
                counter = 0
                if concept not in frequency_counter:
                    print(concept)
                    for file in files:
                        if file == ".DS_Store":
                            continue
                        try:
                            text = open(path + file, "r", encoding="utf-8").read()
                            pattern = r"\b" + concept.lower() + r"\b"
                            counter += len(
                                self.re_match(pattern, text.lower()))  # accumulate the number of the concepts

                        except FileNotFoundError:
                            continue

                    frequency_counter[concept] = counter

        with open("wiki_concept/frequency/frequency_v2.pkt",
                  "wb") as f:
            pickle.dump(frequency_counter, f)

        for i in range(len(df)):
            DC, EC = df.at[i, 'DC'], df.at[i, 'EC']
            df.at[i, 'DC_freq'] = frequency_counter[DC]
            df.at[i, 'EC_freq'] = frequency_counter[EC]

        return df

    def distributional_similarity(self, df, embedding):
        """ This function calculates the distributional similarity between DC and EC"""
        # df.loc[:, 'distributional_similarity'] = 0.0  # initialization
        dc_embedding = []
        ec_embedding = []
        distributional_similarity = []
        # import the class Word2vec from Helfunctions to calculate the word embeddings
        w = Word2vec(df)
        if embedding == 'spacy':
            word2vec = w.embedding_spacy()
        elif embedding == 'fasttext':
            word2vec = w.embedding_fasttext()
        elif embedding == 'statified':
            word2vec = w.embedding_statified()
        else:
            raise Exception('Embedding not found!!')

        for i in range(len(df)):
            DC, EC = df.at[i, 'DC'], df.at[i, 'EC']
            dc_embedding.append(word2vec[DC].tolist())
            ec_embedding.append(word2vec[EC].tolist())
            distributional_similarity.append(
                w.cos_similarity(word2vec[DC], word2vec[EC]))  # calculate cosine similarity
        df['DC_embedding'] = dc_embedding
        df['EC_embedding'] = ec_embedding
        df['distributional_similarity'] = distributional_similarity

        return df

    def substring(self, df):
        """ EC cannot be the substring of DC or vice versa"""
        df.loc[:, 'substring'] = True
        for i in range(len(df)):
            DC, EC = df.at[i, 'DC'], df.at[i, 'EC']
            if DC.lower() not in EC and DC not in EC and EC not in DC and EC.lower() not in DC:  # if EC not a substring of DC and vice versa
                df.at[i, 'substring'] = False

        return df

    def processing(self, embedding, df):
        """ This function processes all the filters at once"""
        preprocess = self.preprocess(df)
        stop_word = self.stop_word(preprocess)
        substring = self.substring(stop_word)
        #named_entity = self.named_entity(substring)
        dsimilarity = self.distributional_similarity(substring, embedding)
        frequency = self.frequency_ratio(dsimilarity, "wiki_concept/frequency/frequency_v2.pkt")
        #frequency.to_csv("wiki_concept/filter/filter_v1.csv")

        return frequency

    def filter(self, df, freq=0.01, dsim=0.2, path="wiki_concept/frequency/frequency_v2.pkt"):
        """ This function filters our given input and return only the ones that fit our criteria"""
        #df = pd.read_csv("wiki_concept/filter/filter_v1.csv", index_col=0)
        with open(path, "rb") as f:
            frequency = pickle.load(f)

        for i in range(len(df)):
            DC, EC = df.at[i, 'DC'], df.at[i, 'EC']
            if frequency[DC] == 0 or frequency[EC] == 0:
                df.at[i, 'filter out'] = 1
            # elif not df.at[i, 'stop_words'] and df.at[i, 'ner'] and not df.at[i, 'substring'] \
            #         and df.at[i, 'distributional_similarity'] > dsim and min(frequency[DC] / frequency[EC],
            #                                                                  frequency[EC] / frequency[DC]) > freq:
            elif not df.at[i, 'stop_words'] and not df.at[i, 'substring'] \
                    and df.at[i, 'distributional_similarity'] > dsim and min(frequency[DC] / frequency[EC],
                                                                              frequency[EC] / frequency[DC]) > freq:
                df.at[i, 'filter out'] = 0
            else:
                df.at[i, 'filter out'] = 1
        result = df[df["filter out"] == 0.0]
        result = result.reset_index(drop=True)
        return result

    def filter_statistic(self, df, freq=0.01, dsim=0.3, path="wiki_concept/frequency/frequency_v2.pkt"):
        """ This function filters our given input and return only the ones that fit our criteria"""
        # df = self.processing()
        #df = pd.read_csv("wiki_concept/filter/filter_v1.csv", index_col=0)
        with open(path, "rb") as f:
            frequency = pickle.load(f)
        for key, value in frequency.items():
            if value == 0:
                frequency[key] = 1
        l = len(df)
        print(len(df))
        print('Stop words: ',  len(df[~df['stop_words']])/l)
        print('NER: ', len(df[df['ner']])/l)
        print('Substring: ', len(df[~df['substring']])/l)
        print('Similarity: ', len(df[df['distributional_similarity'] > dsim])/l)
        print('Freq:', len(df[df.apply(
            lambda row: min(frequency[row['DC']] / frequency[row['EC']], frequency[row['EC']] / frequency[row['DC']]) > freq, axis=1)])/l)

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
    df = pd.read_csv("wiki_concept/concept/concept_v1.csv", index_col=0)
    f = Filter_en(df)
    f.filter_statistic()
    # # df = f.processing()
    # #f.frequency_ratio(df)
    # df1 = f.filter()
    #
    #
    #df1.to_csv("wiki_concept/filter/sim=0.2_freq=0.01.csv")

    # path = "wiki_concept/wiki_titles/"
    # files = [f for f in listdir(path) if isfile(join(path, f))]  # get all the files in the folder wiki titles
    # words_length = 0
    #
    # for file in files:
    #     if file == ".DS_Store" or ':' in file:
    #         continue
    #     try:
    #         text = open(path + file, "r", encoding="utf-8").read()
    #         text_length = text.split(' ')
    #         words_length += len(text_length)
    #     except FileNotFoundError:
    #         pass
    # print(words_length)

