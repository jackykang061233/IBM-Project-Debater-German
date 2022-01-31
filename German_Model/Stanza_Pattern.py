# spacy
import pickle

import spacy
# stanza
import stanza
from stanza.server import CoreNLPClient
# basic functions
import pandas as pd
pd.set_option('display.max_columns', 10)  # show at most 10 columns


class Stanza_Pattern:
    """
    A class used to perform the pattern extraction using stanza and spacy

    Attributes
    ----------

    Methods
    -------
    sent_segmentation(text):
        This method performs the sentence segmentation of a text with help of spacy
    tokenization(sentence):
        This method returns result of the tokenization of a sentence with help of spacy
    stanza_processor_sentence(sentence):
        This method returns result of the several NLP tasks on each token of a sentence with help of stanza
    pattern_extraction_text(text, dc, pattern):
        1. Do the sentence segmentation of the text
        2. Add the dc to the regex pattern
        3. Extract the pattern from each sentence
        4. Loop through all the matches and get the EC
        5. check the Genetiv
        6. getthe whole matched sentences
    pattern_extraction_sentence(sentence, pattern):
        This method uses the token-level regular expression to match our desired patterns
    pattern_dc_construction(dc, pattern):
        This method uses the token-level regular expression to match our desired patterns
    clean_ec:
        This method removes nan and duplicate ECs after extracting topics
    process(text):
        This method performs the pattern matching process
    """
    # '([lemma: sein]?) /eine/ /form/ /des|der|von|vom/ '
    # ' /,?/ /oder/ /andere/ /Arten/ /des|der|von|vom/'
    def __init__(self):
        """
        Parameters
        ----------
        client:
            a Stanford CoreNLP server of the german language
        pattern:
            all to match patterns
        nlp_spacy:
            the spacy german model
        nlp_stanza:
            the stanza german pipeline
        """
        self.client = CoreNLPClient(properties='german', annotators=['tokenize', 'ssplit', 'pos', 'lemma', 'ner', 'parse'], timeout=30000, memory='16G')
        self.pattern = ['/ist/ /ein|eine/',
                        '/ist/ /eine/ /Art/ /des|der|von|vom/',
                        '/ist/ /eine/ /Form/ /des|der|von|vom/',
                        '/ist/ /ein/ /Beispiel/ /für/',
                        '/ist/ /ein/ /Spezialfall/ /des|der|von|vom/',
                        '/oder/ /andere/ /Arten/ /des|der|von|vom/',
                        '/oder/ /eine/ /andere/ /Art/ /des|der|von|vom/',
                        '/oder/ /ander|andere|anderes/',
                        '/und/ /ander|andere|anderes/',
                        '/und/ /andere/ /Arten/ /des|der|von|vom/'
                        # '/einschließlich',
                        # '/inklusive'
                        ]
        self.nlp_spacy = spacy.load("de_core_news_sm")
        self.nlp_stanza = stanza.Pipeline(lang='de', processors='tokenize, mwt, lemma, pos, depparse')

    def sent_segmentation(self, text):
        """
        This method performs the sentence segmentation of a text with help of spacy

        Parameters
        ----------
        text: String
            the string of a text

        Returns
        ------
        List:
            a list contains of the result of the sentence segmentation of a text

        """
        sentences = []
        doc = self.nlp_spacy(text)
        assert doc.has_annotation("SENT_START")
        for sent in doc.sents:
            sentences.append(sent.text)  # append sentence to list
        return sentences

    def tokenization(self, sentence):
        """
        This method returns result of the tokenization of a sentence with help of spacy

        Parameters
        ----------
        sentence: String
            the string of a sentence

        Returns
        ------
        List:
            a list contains of the result of the tokenization of a sentence

        """
        doc = self.nlp_spacy(sentence)

        return [token.text for token in doc]

    def stanza_processor_sentence(self, sentence):
        """
        This method returns result of the several NLP tasks on each token of a sentence with help of stanza

        Parameters
        ----------
        sentence: String
           the string of a sentence

        Returns
        ------
        List:
           a list contains of the result of stanza pipeline with processors: tokenizer, multi-word tokenizer, lemmatizer, pos tagger, dependency parser

       """
        doc = self.nlp_stanza(sentence)

        return doc.sentences

    def pattern_extraction_text(self, sentences, dc, pattern):
        """
        This method consists of several parts:
        1. Do the sentence segmentation of the text
        2. Add the dc to the regex pattern
        3. Extract the pattern from each sentence
        4. Loop through all the matches and get the EC
        5. check the Genetiv
        6. get the whole matched sentences

        Parameters
        ----------
        text: String
            the string of text
        dc: String
            debate concept
        pattern: String
            to match pattern

        Returns
        ------
        List:
            a list contains of every pair of matched (dc, ec, whole sentence)

        """
        pattern_dc = self.pattern_dc_construction(dc, pattern)  # concatenate dc and pattern

        list_rows = []  # store the every pair of (dc, ec, whole sentence)
        for sent in sentences:
            match = self.pattern_extraction_sentence(sent, pattern_dc)  # find the match from the pattern
            dependency_parse = self.stanza_processor_sentence(sent)  # the dependency parser of the sentence
            for index in range(match["sentences"][0]["length"]):
                begin = match['sentences'][0][str(index)]["begin"]  # get the beginning index of each matched sentence
                end = match['sentences'][0][str(index)]["end"]  # get the end index of each matched sentence

                head = dependency_parse[0].words[end-1].head  # the head of the end index word is our potential EC

                ec = " ".join([dependency_parse[0].words[i].text for i in range(end, head)])  # get EC

                # This part of code will try to determine
                if dependency_parse[0].words[head].feats is not None and dependency_parse[0].words[head].feats[:9] == 'Case=Gen|':
                    for i in range(head, len(dependency_parse[0].words)):
                        ec += ' '
                        ec += dependency_parse[0].words[i].text
                        if dependency_parse[0].words[i].head == head:
                            end = i
                            break

                whole_sentence = [dependency_parse[0].words[i].text for i in range(begin, end)]  # get our whole matched sentence till end
                whole_sentence = " ".join(whole_sentence)
                whole_sentence += ' '
                whole_sentence += ec  # plus the ec

                dict_row = {'DC': dc, 'EC': ec, 'Whole Sentence': whole_sentence}  # put in the dict
                list_rows.append(dict_row)
                # print(dependency_parse)

        return list_rows

    def pattern_extraction_sentence(self, sentence, pattern):
        """
        This method uses the token-level regular expression to match our desired patterns

        Parameters
        ----------
        sentence: String
            the string of a sentence
        pattern: String
            to match pattern

        Returns
        ------
        Dict:
            a dictionary contains of the matched patterns in a sentence
       """
        matches = self.client.tokensregex(sentence, pattern)

        return matches

    def pattern_dc_construction(self, dc, pattern):
        """
        This method returns the string of the concatenation of dc and pattern

        Parameters
        ----------
        dc: String
            debate concept
        pattern: String
            to match pattern

        Returns
        ------
        String
            a string contains of the concatenation of dc and pattern

        """
        tokens = self.tokenization(dc)
        pattern_dc = "/ /".join(tokens)  # add the dc to the regex
        pattern_dc = '/' + tokens[0].lower() + '|' + pattern_dc + '/ ' + pattern  # dc can be the first word

        return pattern_dc

    def clean_ec(self, extracted_sentences):
        """
        This method removes nan and duplicate ECs after extracting topics

        Parameters
        ----------
         dc: String
            debate concept

        """
        dc_ec_list = extracted_sentences.iloc[:, :2].values

        # remove duplicate and nan
        dc_ec_list = list(set([(concept[0], concept[1]) for concept in dc_ec_list if pd.notna(concept[1])]))
        cleaned_sentences = pd.DataFrame.from_records(dc_ec_list, columns=['DC', 'EC'])
        return cleaned_sentences

    def process(self, extracted_sentence_path, extracted_topic_pairs_path):
        """
        This method performs the pattern matching process

        Parameters
        ----------
        extracted_sentence_path: String
           the path of the dictionary that saves the extracted sentences in the previous step

        Returns
        ------
        Dataframe:
           a list contains of all the matched pairs of DC and EC in the form of DC, EC, Whole Sentence

       """
        list_rows = []
        # with open("/Users/kangchieh/Downloads/Bachelorarbeit/wiki_concept/concept_de/concept_wiki_de.pkt", "rb") as f:
        #     extracted_sentences = pickle.load(f)
        with open(extracted_sentence_path, "rb") as f:
            extracted_sentences = pickle.load(f)
        for dc, sentences in extracted_sentences.items():
            for pattern in self.pattern:
                list_rows += self.pattern_extraction_text(sentences, dc, pattern)

        extracted_sentences = pd.DataFrame(list_rows)
        print(extracted_sentences)
        extracted_sentences = self.clean_ec(extracted_sentences)
        extracted_sentences.to_csv(extracted_topic_pairs_path)


if __name__ == '__main__':
    pass
