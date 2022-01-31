# from German_Model import RE_Pattern, Stanza_Pattern, Filter_de, HelpFunctions_de, Training_de
from German_Model.RE_Pattern import Pattern
from German_Model.Stanza_Pattern import Stanza_Pattern
from German_Model.Filter_de import Filter
from German_Model.Training_de import Training, GetFeature


if __name__ == '__main__':
    # parameters
    dc_path = '/Users/kangchieh/Downloads/Bachelorarbeit/wiki_concept/topic.csv'
    corpus_path = '/Users/kangchieh/Downloads/Bachelorarbeit/corpus_de/'
    extracted_sentence_path = '/Users/kangchieh/Downloads/Bachelorarbeit/wiki_concept/concept_de/final.txt'
    extracted_topic_pairs_path = '/Users/kangchieh/Downloads/Bachelorarbeit/wiki_concept/concept_de/final_topic_pairs.txt'
    frequency_dict = "/Users/kangchieh/Downloads/Bachelorarbeit/wiki_concept/frequency_de/final_frequency.pkt"
    sentiment_path = "/Users/kangchieh/Downloads/Bachelorarbeit/wiki_concept/sentiment/sentiment_de.pkt"

    wiki_cat = "/Users/kangchieh/Downloads/Bachelorarbeit/wiki_concept/wiki/cat_de.pkt"
    wiki_link = "/Users/kangchieh/Downloads/Bachelorarbeit/wiki_concept/wiki/link_de.pkt"

    label_de = "/Users/kangchieh/Downloads/Bachelorarbeit/wiki_concept/expansion_label_de.csv"
    label_en = "/Users/kangchieh/Downloads/Bachelorarbeit/wiki_concept/expansion_label_en.csv"

    # Extract sentences containing pattern
    re_pattern = Pattern(dc_path)
    re_pattern.extract_concept(corpus_path, extracted_sentence_path)

    # Extract topic pairs and save as dataframe
    stanza_pattern = Stanza_Pattern()
    stanza_pattern.process(extracted_sentence_path, extracted_topic_pairs_path)

    # Filter
    filter = Filter()
    de = filter.processing('fasttext', filter.df_de, corpus_path, frequency_dict)
    de = filter.filter(de, dict_freq_path)

    # FEATURE
    feature = GetFeature()
    feature_de = feature.processing(de, feature.german_sentiment)

    # TRAINING
    training = Training(feature_de, False, 'de', label_de, label_en)
    training.process_grid_search(feature_de, 'de', 'logistic', 'de')






