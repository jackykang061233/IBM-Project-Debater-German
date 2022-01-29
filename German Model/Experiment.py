import pickle

import pandas as pd

from Training_de import Training, GetFeature
from Filter_de import Filter

from English_Model.Filter import Filter_en
from English_Model.Training import GetFeature_en
from English_Model.Statified_Word2vec import Statified_Word2vec

pd.set_option('display.max_rows', 100)


class Experiment:
    """
    A class used to do the experiment and compare the result

    Attributes
    ----------
    df_en:
        a dataframe of english topic expansion pairs
    df_de:
        a dataframe of german topic expansion pairs
    german_sentiment:
        a dictionary of german sentiment, if the word already exists in the dictionary then load its sentiment
        if not then process the sentiment function defined in class GetFeature
    english_sentiment:
        a dictionary of english sentiment, if the word already exists in the dictionary then load its sentiment
        if not then process the sentiment function defined in class GetFeature_en
    Methods
    -------
    logistic_vs_decision_de:
        This method compares the result of using logistic regression and decision trees as training model
    statified_contextual_vs_fasttext_de:
        This method compares the result of using the german version of statified contextualized word embedding and
        fasttext as word embedding
    statified_contextual_vs_fasttext:
        This method compares the result of using the english version of statified contextualized word embedding or
        fasttext as word embedding
    en_de_final:
        This method compares the result of the final version of english and german model
    translation_extraction_first_compare:
        This method compares the result of translating first and extract patterns first
    """

    def __init__(self, df_en=None, df_de=None, german_sentiment=None, english_sentiment=None):
        self.df_en = df_en
        self.df_de = df_de
        self.german_sentiment = german_sentiment
        self.english_sentiment = english_sentiment

    def logistic_vs_decision_de(self):
        """This method compares the result of using logistic regression or decision trees as training model"""
        # FILTER
        filter = Filter()
        de = filter.processing('fasttext', self.df_de)
        de = filter.filter(de)

        # FEATURE
        feature = GetFeature()
        # "/Users/kangchieh/Downloads/Bachelorarbeit/wiki_concept/wiki/cat_de.pkt","/Users/kangchieh/Downloads/Bachelorarbeit/wiki_concept/wiki/link_de.pkt"
        feature_de = feature.processing(de, self.german_sentiment)

        # TRAINING
        training = Training(feature_de, False, 'de')
        # training.f1_score_compare(df=feature_de)
        training.process_grid_search(feature_de, 'de', 'logistic', 'logistic')
        training.process_grid_search(feature_de, 'de', 'decision', 'decision')

    def statified_contextual_vs_fasttext_de(self):
        """This method compares the result of using the german version of statified contextualized word embedding or
        fasttext as word embedding"""
        # FILTER
        filter = Filter()
        statified = filter.processing('statified', self.df_de)
        fasttext = filter.processing('fasttext', self.df_de)

        statified_filter = filter.filter(statified)
        fasttext_filter = filter.filter(fasttext)

        # FEATURE
        feature = GetFeature()

        feature_statified = feature.processing(statified_filter, self.german_sentiment)
        feature_fasttext = feature.processing(fasttext_filter, self.german_sentiment)

        # TRAINING
        training = Training(feature_statified, False, 'de')
        training.process_grid_search(feature_statified, 'de', 'logistic', 'statified')
        training.process_grid_search(feature_fasttext, 'de', 'logistic', 'fasttext')
        # training.f1_score_compare(lang='de', df=feature_fasttext)

    def statified_contextual_vs_fasttext(self):
        """This method compares the result of using the english version of statified contextualized word embedding or
        fasttext as word embedding"""
        # FILTER
        filter = Filter_en()
        statified = filter.processing('statified', self.df_en)
        fasttext = filter.processing('fasttext', self.df_en)

        statified_filter = filter.filter(statified)
        fasttext_filter = filter.filter(fasttext)

        # FEATURE
        feature = GetFeature_en()

        feature_statified = feature.processing(statified_filter, self.english_sentiment)
        feature_fasttext = feature.processing(fasttext_filter, self.english_sentiment)

        # TRAINING
        feature_statified.to_csv("/Users/kangchieh/Downloads/feature_statified.csv")
        training = Training(feature_statified, False, 'en')
        training.f1_score_compare(lang='en', df=feature_fasttext)

    def en_de_final(self):
        """This method compares the result of the final version of english and german model"""
        # FILTER
        filter_en = Filter_en()
        filter = Filter()
        en = filter_en.processing('fasttext', self.df_en)
        de = filter.processing('fasttext', self.df_de)

        en = filter_en.filter(en)
        de = filter.filter(de)

        # FEATURE
        feature_en = GetFeature_en()
        feature = GetFeature()

        feature_english = feature_en.processing(en, self.english_sentiment)
        feature_de = feature.processing(de, self.german_sentiment)

        # TRAINING

        training = Training(feature_english, False, 'en')
        # training.f1_score_compare(df=feature_de)
        training.process_grid_search(feature_english, 'en', 'logistic', 'en')
        training.process_grid_search(feature_de, 'de', 'logistic', 'de')

    def translation_extraction_first_compare(self):
        """This method compares the result of translating first and extract patterns first"""
        # translate extraction patterns into german
        with open(
                '/Users/kangchieh/Downloads/Bachelorarbeit/wiki_concept/translated/topic_translation_LiberTranslate.txt',
                'rb') as f:
            translate = pickle.load(f)
        translated_df_de = pd.DataFrame(df_en['DC'].apply(lambda x: translate[x]))
        translated_df_de['EC'] = df_en['EC'].apply(lambda x: translate[x])

        # FILTER
        filter = Filter()
        extract = filter.processing('fasttext', translated_df_de)
        translate = filter.processing('fasttext', self.df_de)

        extract_filter = filter.filter(extract)
        translate_filter = filter.filter(translate)

        # FEATURE
        feature = GetFeature()

        feature_extract = feature.processing(extract_filter, self.german_sentiment)
        feature_translate = feature.processing(translate_filter, self.german_sentiment)

        # TRAINING
        training = Training(feature_extract, False, 'de')
        training.process_grid_search(feature_extract, 'de', 'logistic', 'extract')
        training.process_grid_search(feature_translate, 'de', 'logistic', 'translate')
    # def distributional_similarity_translation(self):
    #     fairseq = self.df_en
    #     libre = self.df_en
    #     #  TRANSLATION
    #     with open("/Users/kangchieh/Downloads/Bachelorarbeit/wiki_concept/translated/topic_translation_fairseq.txt",
    #               "rb") as f:
    #         fairseq_dict = pickle.load(f)
    #     with open(
    #             "/Users/kangchieh/Downloads/Bachelorarbeit/wiki_concept/translated/topic_translation_LiberTranslate.txt",
    #             "rb") as f:
    #         libre_dict = pickle.load(f)
    #
    #     # FILTER
    #     filter = Filter()
    #     fairseq['DC'].replace(fairseq_dict, inplace=True)
    #     fairseq['EC'].replace(fairseq_dict, inplace=True)
    #     libre['DC'].replace(libre_dict, inplace=True)
    #     libre['EC'].replace(libre_dict, inplace=True)
    #
    #     fairseq_filter = filter.processing('fasttext', fairseq)
    #     libre_filter = filter.processing('fasttext', libre)
    #
    #     fairseq_filter = filter.filter(fairseq_filter)
    #     libre_filter = filter.filter(libre_filter)
    #
    #     # FEATURE
    #     feature = GetFeature()
    #
    #     feature_fairseq = feature.processing(fairseq_filter,
    #                                        "/Users/kangchieh/Downloads/Bachelorarbeit/wiki_concept/sentiment/sentiment_de.pkt",
    #                                        "/Users/kangchieh/Downloads/Bachelorarbeit/wiki_concept/wiki/cat_de.pkt",
    #                                        "/Users/kangchieh/Downloads/Bachelorarbeit/wiki_concept/wiki/link_de.pkt")
    #     feature_libre = feature.processing(libre_filter,
    #                                           "/Users/kangchieh/Downloads/Bachelorarbeit/wiki_concept/sentiment/sentiment_de.pkt",
    #                                           "/Users/kangchieh/Downloads/Bachelorarbeit/wiki_concept/wiki/cat_de.pkt",
    #                                           "/Users/kangchieh/Downloads/Bachelorarbeit/wiki_concept/wiki/link_de.pkt")
    #
    #     # TRAINING
    #     training = Training(feature_fairseq, False, 'en')
    #     training.f1_score_compare(df=feature_libre)

    # def distributional_similarity(self):
    #     # FILTER
    #     filter = Filter()
    #     sim_02 = filter.processing('fasttext', self.df_de)
    #     sim_03 = filter.processing('fasttext', self.df_de)
    #
    #     sim_02_filter = filter.filter(sim_02, dsim=0.2)
    #     sim_03_filter = filter.filter(sim_03)
    #
    #     # FEATURE
    #     feature = GetFeature()
    #
    #     feature_sim_02 = feature.processing(sim_02_filter, "/Users/kangchieh/Downloads/Bachelorarbeit/wiki_concept/sentiment/sentiment_de.pkt", "/Users/kangchieh/Downloads/Bachelorarbeit/wiki_concept/wiki/cat_de.pkt", "/Users/kangchieh/Downloads/Bachelorarbeit/wiki_concept/wiki/link_de.pkt")
    #     feature_sim_03 = feature.processing(sim_03_filter, "/Users/kangchieh/Downloads/Bachelorarbeit/wiki_concept/sentiment/sentiment_de.pkt", "/Users/kangchieh/Downloads/Bachelorarbeit/wiki_concept/wiki/cat_de.pkt", "/Users/kangchieh/Downloads/Bachelorarbeit/wiki_concept/wiki/link_de.pkt")
    #
    #     # TRAINING
    #     training = Training(feature_sim_02, False, 'de')
    #     training.f1_score_compare(df=feature_sim_03)
    #
    # def distributional_similarity_word_embedding(self):
    #     # FILTER
    #     filter = Filter()
    #     spacy = filter.processing('spacy', self.df_de)
    #     fasttext = filter.processing('fasttext', self.df_de)
    #
    #     spacy_filter = filter.filter(spacy)
    #     fasttext_filter = filter.filter(fasttext)
    #
    #     # FEATURE
    #     feature = GetFeature()
    #
    #     feature_spacy = feature.processing(spacy_filter, "/Users/kangchieh/Downloads/Bachelorarbeit/wiki_concept/sentiment/sentiment_de.pkt", "/Users/kangchieh/Downloads/Bachelorarbeit/wiki_concept/wiki/cat_de.pkt", "/Users/kangchieh/Downloads/Bachelorarbeit/wiki_concept/wiki/link_de.pkt")
    #     feature_fasttext = feature.processing(fasttext_filter, "/Users/kangchieh/Downloads/Bachelorarbeit/wiki_concept/sentiment/sentiment_de.pkt", "/Users/kangchieh/Downloads/Bachelorarbeit/wiki_concept/wiki/cat_de.pkt", "/Users/kangchieh/Downloads/Bachelorarbeit/wiki_concept/wiki/link_de.pkt")
    #
    #     # TRAINING
    #     training = Training(feature_spacy, False, 'de')
    #     training.f1_score_compare(df=feature_fasttext)

    # def en_de_without_dsim(self):
    #     filter_en = Filter_en()
    #     filter = Filter()
    #     en = filter_en.processing('fasttext', self.df_en)
    #     de = filter.processing('fasttext', self.df_de)
    #
    #     en = filter_en.filter(en)
    #     de = filter.filter(de)
    #
    #     # FEATURE
    #     feature_en = GetFeature_en()
    #     feature = GetFeature()
    #
    #     feature_english = feature_en.processing(en,
    #                                        "/Users/kangchieh/Downloads/Bachelorarbeit/wiki_concept/sentiment_v1.pkt",
    #                                     "/Users/kangchieh/Downloads/Bachelorarbeit/wiki_concept/wiki/cat_en.pkt",
    #                                     "/Users/kangchieh/Downloads/Bachelorarbeit/wiki_concept/wiki/link_en.pkt"
    #                                        )
    #     feature_de = feature.processing(de,
    #                                           "/Users/kangchieh/Downloads/Bachelorarbeit/wiki_concept/sentiment/sentiment_de.pkt",
    #                                           "/Users/kangchieh/Downloads/Bachelorarbeit/wiki_concept/wiki/cat_de.pkt",
    #                                           "/Users/kangchieh/Downloads/Bachelorarbeit/wiki_concept/wiki/link_de.pkt")
    #
    #     # TRAINING
    #     training = Training(feature_english, False, 'en')
    #     training.f1_score_compare(df=feature_de, drop=['distributional_similarity'])


if __name__ == "__main__":
    # df = pd.read_csv("/Users/kangchieh/Downloads/Bachelorarbeit/wiki_concept/concept_de/concept_wiki_de_v2.csv",
    #                  index_col=0)
    df_en_path = ''
    df_de_path = ''
    df_en = pd.read_csv("/Users/kangchieh/Downloads/Bachelorarbeit/wiki_concept/concept/concept.csv",
                        index_col=0)
    df_de = pd.read_csv("/Users/kangchieh/Downloads/Bachelorarbeit/wiki_concept/expansion_label_de.csv",
                        index_col=0)

    experiment = Experiment(df_en=df_en, df_de=df_de)
    experiment.en_de_final()
    # df_en = pd.read_csv("/Users/kangchieh/Downloads/Bachelorarbeit/wiki_concept/expansion_label_en.csv",
    # index_col=0)

