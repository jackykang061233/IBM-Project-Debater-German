# regular expression
import re
# wikipedia api
import wikipediaapi
# postag and chunk
import spacy
# basic functions
import pandas as pd
import pickle
from os import listdir
from os.path import isfile, join



class Pattern:
    """
    This class is aimed to find the pattern of the potential extraction candidates and extract the EC from the matched
    sentences.
    ...
    Attributes
    ----------
    dc_path:
        the path to a dataframe of DCs and their outlinks
    Methods
    -------
    extract_concept():
        uses the matching patterns to find the potential EC (normally a sentence here)
    re_match(pattern, text):
        uses re to find the matched sentence in our text
    postag(sentence):
        finds the postag of a sentence
    """

    def __init__(self, dc_path):
        """
        Parameters
        ----------
        spacy:
            a small german Spacy pipeline trained on written web text that containing tok2vec, tagger, parser, senter,
            attribute_ruler, lemmatizer, ner
        dc:
            a Dataframe containing our Debate Topics with wiki outlinks
        consistent_pattern :
            a list containing the matching consistent patterns (re)
        """
        self.spacy = spacy.load("de_core_news_sm")  # spacy postag
        self.dc = pd.read_csv(dc_path,  index_col=0)
        self.consistent_pattern = [r' ist (ein|eine))\s+(.+?)\.',
                                   r' ist eine Art (des|der|von|vom))\s+(.+?)\.',
                                   r' ist eine Form (des|der|von|vom))\s+(.+?)\.',
                                   r' ist ein Beispiel für)\s+(.+?)\.',
                                   r' ist ein Spezialfall (des|der|von|vom))\s+(.+?)\.',
                                   r' oder andere Arten (des|der|von|vom))\s+(.+?)\.',
                                   r' oder eine andere Art (des|der|von|vom))\s+(.+?)\.',
                                   r' oder (ander|andere|anderes))\s+(.+?)\.',
                                   r' und (ander|andere|anderes))\s+(.+?)\.',
                                   r' und andere Arten (des|der|von|vom))\s+(.+?)\.']
        # r'((\w+\-*\s*){1,10})\bsuch as ',
        # r'((\w+\-*\s*){1,10})\b(including ',
        # r'((\w+\-*\s*){1,10})\b(e\.g\. ']

    def extract_concept(self, corpus_path, extracted_sentence_path):
        """
        This function uses the matching patterns to find the potential EC (normally a sentence here)

        Parameters
        ----------
        corpus_path: String
            the path of folder of our corpus
        extracted_sentence_path: String
            the path of the dictionary to save to extracted sentences
        """
        topic = {concept: [] for concept in self.dc.index.values}  # get our main topics and create an empty list
        corpus_files = [join(corpus_path, f) for f in listdir(corpus_path) if isfile(join(corpus_path, f))]  # list all corpus files
        for concept in self.dc.index.values:
            print(concept)
            # with open(corpus_path, "rb") as f:
            #     extracted_sentences = pickle.load(f)
            for file in corpus_files:
                with open(file, "r") as f:  # read files one by one
                        lines = f.readlines()
                lines = ' '.join(lines)

                for pattern in self.consistent_pattern:
                    concept_pattern = r'(' + concept + pattern  # create pattern from main topic
                    match = self.re_match(concept_pattern, lines)  # find re match in outlink files
                    if match:
                        for m in match:
                            try:
                                topic[concept].append(
                                    m[0] + ' ' + m[2] + '.')  # add the sentence matched with the pattern
                            except IndexError:
                                topic[concept].append(m[0] + ' ' + m[1] + '.')

            topic[concept] = list(set(topic[concept]))  # delete duplicates

        with open(extracted_sentence_path, "wb") as f:
            pickle.dump(topic, f)

    def re_match(self, pattern, text):
        """
        This functions uses re to find the matched sentence in our text
        Parameters
        ----------
        sentence: String
            a string of sentence
        Returns
        ------
        List:
            a list of postags of the input sentence
        """
        pattern = re.compile(pattern)
        match = pattern.findall(text)
        return match

    def postag(self, sentence):
        """
        This function finds the postag of a sentence

        Parameters
        ----------
        sentence: String
            a string of sentence
        Returns
        ------
        List:
            a list of postags of the input sentence
        """
        pos = []
        sentence = self.spacy(u'' + sentence)
        for word in sentence:
            pos.append(word.pos_)
        return pos


if __name__ == "__main__":
    dc_path = ''
    corpus_path = ''
    saved_extracted_path = ''
    p = Pattern(dc_path)
    p.extract_concept(corpus_path, saved_extracted_path)
