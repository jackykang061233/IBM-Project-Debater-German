import pandas as pd
import numpy as np

from Filter_de import Filter
from HelpFunctions_de import Wordnet, Wiki, Word2vec
from Training_de import GetFeature, Training
from RE_Pattern import Pattern
from Stanza_Pattern import Stanza_Pattern

def func(DC, EC):
    label = pd.read_csv("/Users/kangchieh/Downloads/Bachelorarbeit/wiki_concept/expansion_label_de.csv", index_col=0)

    return label[label.DC == DC & label.EC == EC]['good expansion']

if __name__ == "__main__":
    df = pd.read_csv("/Users/kangchieh/Downloads/Bachelorarbeit/wiki_concept/concept_de/concept_wiki_de_v2.csv",
                     index_col=0)

    # FILTER
    filter = Filter(df)
    filter.processing()
    filter_out = filter.filter()
    filter_out = filter_out.reset_index(drop=True)

    # FEATURE
    Feature = GetFeature(filter_out)
    all_features = Feature.processing()
    all_features = all_features.reset_index(drop=True)

    # Training
    label = pd.read_csv("/Users/kangchieh/Downloads/Bachelorarbeit/wiki_concept/expansion_label_de.csv", index_col=0)
    df1 = all_features.merge(label, on=['DC', 'EC'])

    training = Training(df1)
    training.process(df1)

