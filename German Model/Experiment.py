import pickle

import pandas as pd

from Training_de import Training, GetFeature
from Filter_de import Filter

from English_Model.Filter import Filter_en
from English_Model.Training import GetFeature_en
from English_Model.Statified_Word2vec import Statified_Word2vec
pd.set_option('display.max_rows', 100)

class Experiment:
    def __init__(self, df_en=None, df_de=None):
        self.df_en = df_en
        self.df_de = df_de

    def distributional_similarity(self):
        # FILTER
        filter = Filter()
        sim_02 = filter.processing('fasttext', self.df_de)
        sim_03 = filter.processing('fasttext', self.df_de)

        sim_02_filter = filter.filter(sim_02, dsim=0.2)
        sim_03_filter = filter.filter(sim_03)

        # FEATURE
        feature = GetFeature()

        feature_sim_02 = feature.processing(sim_02_filter, "/Users/kangchieh/Downloads/Bachelorarbeit/wiki_concept/sentiment/sentiment_de.pkt", "/Users/kangchieh/Downloads/Bachelorarbeit/wiki_concept/wiki/cat_de.pkt", "/Users/kangchieh/Downloads/Bachelorarbeit/wiki_concept/wiki/link_de.pkt")
        feature_sim_03 = feature.processing(sim_03_filter, "/Users/kangchieh/Downloads/Bachelorarbeit/wiki_concept/sentiment/sentiment_de.pkt", "/Users/kangchieh/Downloads/Bachelorarbeit/wiki_concept/wiki/cat_de.pkt", "/Users/kangchieh/Downloads/Bachelorarbeit/wiki_concept/wiki/link_de.pkt")

        # TRAINING
        training = Training(feature_sim_02, False, 'de')
        training.f1_score_compare(df=feature_sim_03)

    def distributional_similarity_word_embedding(self):
        # FILTER
        filter = Filter()
        spacy = filter.processing('spacy', self.df_de)
        fasttext = filter.processing('fasttext', self.df_de)

        spacy_filter = filter.filter(spacy)
        fasttext_filter = filter.filter(fasttext)

        # FEATURE
        feature = GetFeature()

        feature_spacy = feature.processing(spacy_filter, "/Users/kangchieh/Downloads/Bachelorarbeit/wiki_concept/sentiment/sentiment_de.pkt", "/Users/kangchieh/Downloads/Bachelorarbeit/wiki_concept/wiki/cat_de.pkt", "/Users/kangchieh/Downloads/Bachelorarbeit/wiki_concept/wiki/link_de.pkt")
        feature_fasttext = feature.processing(fasttext_filter, "/Users/kangchieh/Downloads/Bachelorarbeit/wiki_concept/sentiment/sentiment_de.pkt", "/Users/kangchieh/Downloads/Bachelorarbeit/wiki_concept/wiki/cat_de.pkt", "/Users/kangchieh/Downloads/Bachelorarbeit/wiki_concept/wiki/link_de.pkt")

        # TRAINING
        training = Training(feature_spacy, False, 'de')
        training.f1_score_compare(df=feature_fasttext)

    def distributional_similarity_word_embedding_statified_contextual_vs_static(self):
        # FILTER
        filter = Filter_en()
        statified = filter.processing('statified', self.df_en)
        fasttext = filter.processing('fasttext', self.df_en)

        statified_filter = filter.filter(statified)
        fasttext_filter = filter.filter(fasttext)

        # FEATURE
        feature = GetFeature_en()

        feature_statified = feature.processing(statified_filter,  "/Users/kangchieh/Downloads/Bachelorarbeit/wiki_concept/sentiment_v1.pkt",
                                        "/Users/kangchieh/Downloads/Bachelorarbeit/wiki_concept/wiki/cat_en.pkt",
                                        "/Users/kangchieh/Downloads/Bachelorarbeit/wiki_concept/wiki/link_en.pkt")
        feature_fasttext = feature.processing(fasttext_filter,"/Users/kangchieh/Downloads/Bachelorarbeit/wiki_concept/sentiment_v1.pkt",
                                        "/Users/kangchieh/Downloads/Bachelorarbeit/wiki_concept/wiki/cat_en.pkt",
                                        "/Users/kangchieh/Downloads/Bachelorarbeit/wiki_concept/wiki/link_en.pkt")

        # TRAINING
        feature_statified.to_csv("/Users/kangchieh/Downloads/feature_statified.csv")
        training = Training(feature_statified, False, 'en')
        training.f1_score_compare(lang='en', df=feature_fasttext)

    def distributional_similarity_word_embedding_statified_contextual_vs_static_de(self):
        # FILTER
        filter = Filter()
        statified = filter.processing('statified', self.df_de)
        fasttext = filter.processing('fasttext', self.df_de)

        statified_filter = filter.filter(statified)
        fasttext_filter = filter.filter(fasttext)

        # FEATURE
        feature = GetFeature()

        feature_statified = feature.processing(statified_filter,  "/Users/kangchieh/Downloads/Bachelorarbeit/wiki_concept/sentiment/sentiment_de.pkt", "/Users/kangchieh/Downloads/Bachelorarbeit/wiki_concept/wiki/cat_de.pkt", "/Users/kangchieh/Downloads/Bachelorarbeit/wiki_concept/wiki/link_de.pkt")
        feature_fasttext = feature.processing(fasttext_filter, "/Users/kangchieh/Downloads/Bachelorarbeit/wiki_concept/sentiment/sentiment_de.pkt", "/Users/kangchieh/Downloads/Bachelorarbeit/wiki_concept/wiki/cat_de.pkt", "/Users/kangchieh/Downloads/Bachelorarbeit/wiki_concept/wiki/link_de.pkt")

        # TRAINING
        training = Training(feature_statified, False, 'de')
        training.f1_score_compare(lang='de', df=feature_fasttext)

    def distributional_similarity_translation(self):
        fairseq = self.df_en
        libre = self.df_en
        #  TRANSLATION
        with open("/Users/kangchieh/Downloads/Bachelorarbeit/wiki_concept/translated/topic_translation_fairseq.txt",
                  "rb") as f:
            fairseq_dict = pickle.load(f)
        with open(
                "/Users/kangchieh/Downloads/Bachelorarbeit/wiki_concept/translated/topic_translation_LiberTranslate.txt",
                "rb") as f:
            libre_dict = pickle.load(f)

        # FILTER
        filter = Filter()
        fairseq['DC'].replace(fairseq_dict, inplace=True)
        fairseq['EC'].replace(fairseq_dict, inplace=True)
        libre['DC'].replace(libre_dict, inplace=True)
        libre['EC'].replace(libre_dict, inplace=True)

        fairseq_filter = filter.processing('fasttext', fairseq)
        libre_filter = filter.processing('fasttext', libre)

        fairseq_filter = filter.filter(fairseq_filter)
        libre_filter = filter.filter(libre_filter)

        # FEATURE
        feature = GetFeature()

        feature_fairseq = feature.processing(fairseq_filter,
                                           "/Users/kangchieh/Downloads/Bachelorarbeit/wiki_concept/sentiment/sentiment_de.pkt",
                                           "/Users/kangchieh/Downloads/Bachelorarbeit/wiki_concept/wiki/cat_de.pkt",
                                           "/Users/kangchieh/Downloads/Bachelorarbeit/wiki_concept/wiki/link_de.pkt")
        feature_libre = feature.processing(libre_filter,
                                              "/Users/kangchieh/Downloads/Bachelorarbeit/wiki_concept/sentiment/sentiment_de.pkt",
                                              "/Users/kangchieh/Downloads/Bachelorarbeit/wiki_concept/wiki/cat_de.pkt",
                                              "/Users/kangchieh/Downloads/Bachelorarbeit/wiki_concept/wiki/link_de.pkt")

        # TRAINING
        training = Training(feature_fairseq, False, 'en')
        training.f1_score_compare(df=feature_libre)

    def translation_extraction_first_compare(self):
        pass

    def en_de_without_dsim(self):
        filter_en = Filter_en()
        filter = Filter()
        en = filter_en.processing('fasttext', self.df_en)
        de = filter.processing('fasttext', self.df_de)

        en = filter_en.filter(en)
        de = filter.filter(de)

        # FEATURE
        feature_en = GetFeature_en()
        feature = GetFeature()

        feature_english = feature_en.processing(en,
                                           "/Users/kangchieh/Downloads/Bachelorarbeit/wiki_concept/sentiment_v1.pkt",
                                        "/Users/kangchieh/Downloads/Bachelorarbeit/wiki_concept/wiki/cat_en.pkt",
                                        "/Users/kangchieh/Downloads/Bachelorarbeit/wiki_concept/wiki/link_en.pkt"
                                           )
        feature_de = feature.processing(de,
                                              "/Users/kangchieh/Downloads/Bachelorarbeit/wiki_concept/sentiment/sentiment_de.pkt",
                                              "/Users/kangchieh/Downloads/Bachelorarbeit/wiki_concept/wiki/cat_de.pkt",
                                              "/Users/kangchieh/Downloads/Bachelorarbeit/wiki_concept/wiki/link_de.pkt")

        # TRAINING
        training = Training(feature_english, False, 'en')
        training.f1_score_compare(df=feature_de, drop=['distributional_similarity'])

    def en_de_normal(self):
        filter_en = Filter_en()
        filter = Filter()
        en = filter_en.processing('fasttext', self.df_en)
        de = filter.processing('fasttext', self.df_de)

        en = filter_en.filter(en)
        de = filter.filter(de)

        # FEATURE
        feature_en = GetFeature_en()
        feature = GetFeature()

        feature_english = feature_en.processing(en,
                                                "/Users/kangchieh/Downloads/Bachelorarbeit/wiki_concept/sentiment_v1.pkt",
                                                "/Users/kangchieh/Downloads/Bachelorarbeit/wiki_concept/wiki/cat_en.pkt",
                                                "/Users/kangchieh/Downloads/Bachelorarbeit/wiki_concept/wiki/link_en.pkt"
                                                )
        feature_de = feature.processing(de,
                                        "/Users/kangchieh/Downloads/Bachelorarbeit/wiki_concept/sentiment/sentiment_de.pkt",
                                        "/Users/kangchieh/Downloads/Bachelorarbeit/wiki_concept/wiki/cat_de.pkt",
                                        "/Users/kangchieh/Downloads/Bachelorarbeit/wiki_concept/wiki/link_de.pkt")

        # TRAINING

        training = Training(feature_english, False, 'en')
        training.f1_score_compare(df=feature_de)


if __name__ == "__main__":
    df = pd.read_csv("/Users/kangchieh/Downloads/Bachelorarbeit/wiki_concept/concept_de/concept_wiki_de_v2.csv",
                     index_col=0)
    df_en = pd.read_csv("/Users/kangchieh/Downloads/Bachelorarbeit/wiki_concept/concept/concept.csv",
                     index_col=0)

    experiment = Experiment(df_en=df_en, df_de=df)

    # experiment.distributional_similarity_word_embedding()
    #experiment.distributional_similarity()
    #experiment.distributional_similarity_translation()
    #experiment.en_de_without_dsim()
    #experiment.en_de_normal()
    experiment.distributional_similarity_word_embedding_statified_contextual_vs_static_de()

