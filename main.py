# from German_Model import RE_Pattern, Stanza_Pattern, Filter_de, HelpFunctions_de, Training_de
from German_Model.RE_Pattern import Pattern
from German_Model.Stanza_Pattern import Stanza_Pattern
from German_Model.Filter_de import Filter
from German_Model.Feature_de import GetFeature
from Training import Training

if __name__ == '__main__':
    # PARAMETERS
    # RE_Pattern
    dc_path = 'topic.csv'
    corpus_path = 'corpus_de/'
    extracted_sentence_path = 'final.txt'
    # Stanza_Pattern
    extracted_topic_pairs_path = 'final_topic_pairs.csv'
    # Filter
    frequency_dict = "final_frequency.pkt"
    sentiment_path = "sentiment_de.pkt"

#     wiki_cat = "/Users/kangchieh/Downloads/Bachelorarbeit/wiki_concept/wiki/cat_de.pkt"
#     wiki_link = "/Users/kangchieh/Downloads/Bachelorarbeit/wiki_concept/wiki/link_de.pkt"
    
    # Training
    label_de = "expansion_label_de.csv"
    label_en = "expansion_label_en.csv"
    
    
    # PROCESSING
    # Extract sentences containing pattern
    re_pattern = Pattern(dc_path)
    re_pattern.extract_concept(corpus_path, extracted_sentence_path)
    
    # Extract topic pairs and save as dataframe
    stanza_pattern = Stanza_Pattern()
    stanza_pattern.process(extracted_sentence_path, extracted_topic_pairs_path)

    # Filter
    filter = Filter()
    de = filter.processing('fasttext', extracted_topic_pairs_path, corpus_path, frequency_dict)
    de = filter.filter(de, frequency_dict)

    # FEATURE
    feature = GetFeature()
    feature_de = feature.processing(de, sentiment_path)

    # TRAINING
    training = Training(feature_de, 'de', label_de, label_en)
    training.process_grid_search(feature_de, 'de', 'logistic')






