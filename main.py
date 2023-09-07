# from German_Model import RE_Pattern, Stanza_Pattern, Filter_de, HelpFunctions_de, Training_de
from German_Model.RE_Pattern import Pattern
from German_Model.Stanza_Pattern import Stanza_Pattern
from German_Model.Filter_de import Filter
from German_Model.Feature_de import GetFeature
from Training import Training

import pandas as pd
if __name__ == '__main__':
    ### PARAMETERS ###
    # RE_Pattern
    dc_path = 'German_Model/example_files/topic.csv'
    corpus_path = 'PATH TO CORPUS (SEE README.md)'
    extracted_sentence_path = 'PATH TO SAVE THE FILE'
    
    # Stanza_Pattern
    extracted_topic_pairs_path = 'German_Model/example_files/concept_de.csv'
   
    # Filter
    frequency_dict = "German_Model/example_files/final_frequency.pkt"
    sentiment_path = "German_Model/example_files/sentiment_de.pkt"
    embedding_path = "German_Model/example_files/german_embedding.txt"  # statified
    #embedding_path = "PATH TO FASTTEXT EMBEDDING (SEE README.md to download)"  # fasttext
    
    germanet_path = "germanet/GN_V160/GN_V160_XML"  # PLEAS ADJUST TO YOUR PATH

    # Training
    label_de = "Label/expansion_label_de.csv"
    label_en = "Label/expansion_label_en.csv"
    

    ### PROCESSING ###
    # Extract sentences containing pattern
    # re_pattern = Pattern(dc_path)
    # re_pattern.extract_concept(corpus_path, extracted_sentence_path)
    # #
    # # # Extract topic pairs and save as dataframe
    # stanza_pattern = Stanza_Pattern()
    # stanza_pattern.process(extracted_sentence_path, extracted_topic_pairs_path)

    # Filter
    filter = Filter()
    de = filter.processing('statified', extracted_topic_pairs_path, corpus_path, frequency_dict, embedding_path)
    de = filter.filter(de, frequency_dict)

    # FEATURE
    feature = GetFeature()
    feature_de = feature.processing(de, germanet_path, sentiment_path)

    # TRAINING
    training = Training(feature_de, 'de', label_de, label_en)
    training.process_grid_search(feature_de, 'de', 'logistic')






