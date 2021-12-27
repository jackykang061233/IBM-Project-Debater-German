# regular expression
import re
# os
# NER

import spacy
# word2vec
import string

from HelpFunctions_de import Word2vec
# stopwords
from nltk.corpus import stopwords
# basic functions
import pickle
import pandas as pd


class Filter:
    def __init__(self):
        self.stop_words = set(stopwords.words('german'))  # english stopwords

    def preprocess(self, df):
        df = df[df['EC'].notna()]
        df = df.reset_index(drop=True)
        for i in range(len(df)):
            df.at[i, 'EC'] = df.at[i, 'EC'].translate(str.maketrans('', '', string.punctuation))
            df.at[i, 'DC'], df.at[i, 'EC'] = ' '.join(df.at[i, 'DC'].split()), ' '.join(df.at[i, 'EC'].split())

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
        NER = spacy.load("de_core_news_sm")
        df.loc[:, 'ner'] = False
        for i in range(len(df)):
            EC = df.at[i, 'EC']
            if not NER(EC.lower()).ents:  # if not NER exists in EC
                df.at[i, 'ner'] = True

        return df

    def frequency_ratio(self, df, dict_freq=None):
        """ This function calculates the frequency of a word"""
        Concept = list(set(list(df.DC.values) + list(df.EC.values)))
        if dict_freq == None:  # no existing frequency dictionary
            df['DC_freq'] = 0
            df['EC_freq'] = 0
            frequency_counter = {concept: 0 for concept in Concept}

            for concept in Concept:
                counter = 0
                for i in range(1, 20586):
                    try:
                        with open("/Users/kangchieh/Downloads/Bachelorarbeit/corpus_de/test_%s.txt" % i, 'r') as f:
                            lines = f.readlines()
                        lines = ' '.join(lines)
                        pattern = r"\b" + concept + r"\b"
                        counter += len(self.re_match(pattern, lines))

                    except FileNotFoundError:
                        continue

                frequency_counter[concept] += counter

        else:  # uses existing frequency dictionary
            with open(dict_freq, "rb") as f:
                frequency_counter = pickle.load(f)
            for concept in Concept:
                counter = 0
                if concept not in frequency_counter:
                    print(concept)
                    for i in range(1, 20586):
                        try:
                            with open("/Users/kangchieh/Downloads/Bachelorarbeit/corpus_de/test_%s.txt" % i, 'r') as f:
                                lines = f.readlines()
                            lines = ' '.join(lines)
                            pattern = r"\b" + concept + r"\b"
                            counter += len(self.re_match(pattern, lines))

                        except FileNotFoundError:
                            continue

                    frequency_counter[concept] = counter

        for i in range(len(df)):
            DC, EC = df.at[i, 'DC'], df.at[i, 'EC']
            df.at[i, 'DC_freq'] = frequency_counter[DC]
            df.at[i, 'EC_freq'] = frequency_counter[EC]

        with open("/Users/kangchieh/Downloads/Bachelorarbeit/wiki_concept/frequency_de/frequency_v2.pkt",
                  "wb") as f:
            pickle.dump(frequency_counter, f)

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
            dc_embedding.append(list(word2vec[DC]))
            ec_embedding.append(list(word2vec[EC]))
            distributional_similarity.append(w.cos_similarity(word2vec[DC],word2vec[EC]))  # calculate cosine similarity
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
        # named_entity = self.named_entity(substring)
        dsimilarity = self.distributional_similarity(substring, embedding)

        frequency = self.frequency_ratio(dsimilarity,
                                         "/Users/kangchieh/Downloads/Bachelorarbeit/wiki_concept/frequency_de/frequency_v2.pkt")
        #frequency.to_csv("/Users/kangchieh/Downloads/Bachelorarbeit/wiki_concept/filter_de/filter_%s.pkt" % embedding)

        return frequency

    def filter(self, df, freq=0.01, dsim=0.2,
               freq_path="/Users/kangchieh/Downloads/Bachelorarbeit/wiki_concept/frequency_de/frequency_v2.pkt"):
        """ This function filters our given input and return only the ones that fit our criteria"""
        # df = pd.read_csv("/Users/kangchieh/Downloads/Bachelorarbeit/wiki_concept/filter_de/filter_v4.csv",
        #                  index_col=0)
        with open(freq_path, "rb") as f:
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
        #result.to_csv("/Users/kangchieh/Downloads/Bachelorarbeit/wiki_concept/filter_de/filter_sim=%s_freq=%s.csv" % (dsim, freq))
        return result

    def filter_statistic(self, freq=0.01, dsim=0.3,
                         freq_path="/Users/kangchieh/Downloads/Bachelorarbeit/wiki_concept/frequency_de/frequency_v2.pkt"):
        """ This function filters our given input and return only the ones that fit our criteria"""
        df = pd.read_csv("/Users/kangchieh/Downloads/Bachelorarbeit/wiki_concept/filter_de/filter_v4.csv", index_col=0)
        with open(freq_path, "rb") as f:
            frequency = pickle.load(f)
        for key, value in frequency.items():
            if value == 0:
                frequency[key] = 1
        l = len(df)
        print(len(df))
        print(len(df[~df['stop_words']]) / l)
        # print(len(df[df['ner']]) / l)
        print(len(df[~df['substring']]) / l)
        print(len(df[df['distributional_similarity'] > dsim]) / l)
        print(len(df[df.apply(
            lambda row: min(frequency[row['DC']] / frequency[row['EC']],
                            frequency[row['EC']] / frequency[row['DC']]) > freq, axis=1)]) / l)

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
    # t = LibreTranslateAPI(url='https://translate.argosopentech.com/')
    # #t = LibreTranslateAPI(url='https://translate.mentality.rip/')
    # path = "/Users/kangchieh/Downloads/Bachelorarbeit/corpus_de/"
    # files = [f for f in listdir(path) if isfile(join(path, f))]
    # not_exist = []
    # for i in range(1, 20586):
    #     try:
    #         text = open("/Users/kangchieh/Downloads/Bachelorarbeit/corpus_de/test_%s.txt" % i, "r",
    #                     encoding="utf-8").read()
    #     except:
    #         not_exist.append(i)
    # a = [n for n in not_exist if n > 4135]
    # # b = [n for n in a if n >= 7350]
    # # not_exist = reversed(not_exist)
    #
    # for i in a:
    #     print(i)
    #     sentence_length = 0
    #     sentence = ''
    #     translate = ''
    #     try:
    #         with open("/Users/kangchieh/Downloads/Bachelorarbeit/corpus/test_%s.txt" % i, 'r') as f:
    #             lines = f.readlines()
    #
    #         sentences = t.sent_segmentation(lines[0])
    #         for sent in sentences:
    #             sentence_length += len(sent)
    #             sentence += sent
    #             if sentence_length > 20000:
    #                 translate += t.translate(sentence)
    #                 sentence_length = 0
    #                 sentence = ''
    #
    #         translate += t.translate(sentence)
    #
    #         with open("/Users/kangchieh/Downloads/Bachelorarbeit/corpus_de/test_%s.txt" % i, "w",
    #                   encoding="utf-8") as f:
    #             f.write(translate)
    #
    #     except HTTPError:
    #         print('Error: %s' % i)
    #         with open("/Users/kangchieh/Downloads/Bachelorarbeit/exception_11.txt", "a",
    #                   encoding="utf-8") as exception:
    #             exception.write(str(i) + '\n')
    df = pd.read_csv("/Users/kangchieh/Downloads/Bachelorarbeit/wiki_concept/filter/filter_v1.csv",
                     index_col=0)
    with open("/Users/kangchieh/Downloads/Bachelorarbeit/wiki_concept/translated/topic_translation_fairseq.txt",
              "rb") as f:
        fairseq_dict = pickle.load(f)
    with open("/Users/kangchieh/Downloads/Bachelorarbeit/wiki_concept/frequency_de/frequency_v2.pkt",
              "rb") as f:
        a = pickle.load(f)

    df['DC'].replace(fairseq_dict, inplace=True)
    df['EC'].replace(fairseq_dict, inplace=True)
    count = 0
    for i, v in enumerate(df['DC']):
        if v not in a:
            count += 1
    print(count)
    # f.processing()
    # a = f.filter()
    # a = a.reset_index(drop=True)
    # a.to_csv("/Users/kangchieh/Downloads/Bachelorarbeit/wiki_concept/filter_de/filter_sim=0.3_freq=0.01_libre_shorten.csv")
    #

    # f.frequency_ratio(df)
    # df1 = f.filter()

    # for file in files:
    #     print(file)
    #     min_length = float('inf')
    #     if file == ".DS_Store" or ':' in file:
    #         continue
    #     try:
    #         text = open(path + file, "r", encoding="utf-8").read()
    #         # text_length = text.split(' ')
    #         for p in possible_list:
    #             text_remove = text.split(p, 1)[0]
    #             min_length = min(len(text_remove), min_length)
    #         min_length = min(len(text), min_length)
    #         words_length += min_length
    #     except FileNotFoundError:
    #         pass
    # print(words_length)

# 10000 17500 15000 12500 20000 7500 5000 2500
