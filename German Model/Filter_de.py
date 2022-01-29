# Regular expression
import re
import spacy
# Word2vec
import string

from HelpFunctions_de import Word2vec
# Stopwords
from nltk.corpus import stopwords
# Basic functions
import pickle
import pandas as pd


class Filter:
    """
    A class used to filter out the topic pairs which do not meet our requirement

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
        """
        Parameters
        ----------
        spacy:

        """
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
        print(len(frequency[frequency['distributional_similarity']>0.3]))
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

    def count_occurrences(self, word, sentence):
        return sentence.lower().count(word)

    def re_match(self, pattern, text):
        """ This functions uses re to find the matched pattern in our text"""
        pattern = re.compile(pattern)
        match = pattern.findall(text)
        return match


if __name__ == "__main__":
    f = Filter()