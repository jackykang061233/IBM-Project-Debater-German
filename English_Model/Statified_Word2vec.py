import pickle

import numpy as np
import pandas as pd
import stanza

class Statified_Word2vec:
    def __init__(self):
        self.nlp = stanza.Pipeline(lang='en', processors='tokenize', tokenize_no_ssplit=True)
        self.embedding = self.load_embedding()
        self.tokens = self.tokenization()

    def load_embedding(self):
        with open("/Users/kangchieh/Downloads/embedding_en.txt", "rb") as f:
            embedding = pickle.load(f)
        return embedding

    def tokenization(self):
        with open("/Users/kangchieh/Downloads/Bachelorarbeit/wiki_concept/en_topic_tokenization.txt", 'rb') as f:
             tokenization = pickle.load(f)
        return tokenization
        # doc = self.nlp(sentence)
        # for sent in doc.sentences:
        #     return [token.text for token in sent.tokens]

    def word2vec(self, sentence):
        # sentence = sentence.replace('-', ' - ')
        # sentence = sentence.replace("'s", " 's")
        # sentence = sentence.lower()
        tokens = self.tokens[sentence]
        #return np.minimum.reduce([np.array(self.embedding[token]) for token in tokens])
        #return np.maximum.reduce([np.array(self.embedding[token]) for token in tokens])
        return np.mean([np.array(self.embedding[token.lower()]) for token in tokens], axis=0)
        # return embedding / len(tokens)

if __name__ == '__main__':

    # df = pd.read_csv("/Users/kangchieh/Downloads/Bachelorarbeit/wiki_concept/expansion_label_en.csv", index_col=0)
    # topics = list(set(list(df.DC.values) + list(df.EC.values)))
    # tokenization = {}
    # import spacy
    #
    # nlp = spacy.load("en_core_web_sm")
    # for topic in topics:
    #     doc = nlp(topic)
    #     tokenization[topic] = [token.text for token in doc]

    ######################
    # with open("/Users/kangchieh/Downloads/Bachelorarbeit/wiki_concept/en_topic_tokenization.txt", 'rb') as f:
    #       tokenization = pickle.load(f)
    # token_values = list(tokenization.values())
    # token_values = list(set([token for tokens in token_values for token in tokens]))
    #
    # from gensim.models import KeyedVectors
    #
    # model = KeyedVectors.load_word2vec_format("/Users/kangchieh/Downloads/roberta_12layer_para.vec")
    # word_dict = {}
    # for word in token_values:
    #     try:
    #         word_dict[word] = model[word]
    #         print(word)
    #     except:
    #         pass
    ######################
    with open("/Users/kangchieh/Downloads/Bachelorarbeit/wiki_concept/en_topic_tokenization.txt", 'rb') as f:
         tokenization = pickle.load(f)
    v = list(set([value for values in tokenization.values() for value in values]))
    print(v)
    with open("/Users/kangchieh/Downloads/embedding_statified.txt", 'rb') as f:
        word_dict = pickle.load(f)
    count = 0
    for key in v:
        if key.lower() not in word_dict.keys():
            print(key)
            count += 1
    print(count / len(v))

