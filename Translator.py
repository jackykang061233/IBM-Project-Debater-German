import pickle
from os import listdir
from os.path import isfile, join
import re

import fasttext
import pandas as pd
from deep_translator import LingueeTranslator
import wikipediaapi
import spacy

import torch

import json
import sys
import timeit
from urllib import request, parse
from urllib.error import HTTPError

from scipy import spatial


class Translate:
    def __init__(self):
        self.nlp_spacy = spacy.load("en_core_web_sm")
        self.libretranslate = LibreTranslateAPI(api_key='0b7c26f6-49ab-428e-b593-4085c1a00ec7')

    def topic_translation(self, path):
        df = pd.read_csv(path, index_col=0)
        with open("/Users/kangchieh/Downloads/Bachelorarbeit/wiki_concept/translated/topic_translation_LiberTranslate.txt", "rb") as f:
            translated_topic_pairs = pickle.load(f)
        topics = list(set(list(df.DC.values) + list(df.EC.values)))
        for topic in topics:
            if topic not in translated_topic_pairs:
                translated_topic = self.libretranslate.translate(topic)
                translated_topic_pairs[topic] = translated_topic
        with open("/Users/kangchieh/Downloads/Bachelorarbeit/wiki_concept/translated/topic_translation_LiberTranslate.txt", "wb") as f:
            pickle.dump(translated_topic_pairs, f)

    def translation(self):
        path = "/Users/kangchieh/Downloads/Bachelorarbeit/match_sentences/en_shorten.txt"
        with open(path, 'rb') as f:
            text = pickle.load(f)

        translated_sentences = []
        to_translate = ''
        for index, sent in enumerate(text):
            print(index)
            if index <= 180000:
                continue

            if len(to_translate + sent) > 20000:
                translated_sentence = self.libretranslate.translate(to_translate)
                translated_sentences.append(translated_sentence)
                to_translate = ''
            to_translate += sent

        translated_sentence = self.libretranslate.translate(to_translate)
        translated_sentences.append(translated_sentence)
        with open("/Users/kangchieh/Downloads/Bachelorarbeit/wiki_concept/translated/match_sentence_8.txt", "wb") as f:  # first extract pattern then translated
            pickle.dump(translated_sentences, f)



class LibreTranslateAPI:
    """Connect to the LibreTranslate API"""

    """Example usage:
    from argostranslate.apis import LibreTranslateAPI
    lt = LibreTranslateAPI("https://translate.astian.org/")
    print(lt.detect("Hello World"))
    print(lt.languages())
    print(lt.translate("LibreTranslate is awesome!", "en", "es"))
    """

    def __init__(self, url="https://translate.argosopentech.com/", api_key=None):
        """Create a LibreTranslate API connection.

        Args:
            url (str): The url of the LibreTranslate endpoint.
            api_key (str): The API key.
        """

        self.url = url
        self.api_key = api_key
        self.nlp_spacy = spacy.load("en_core_web_sm")

        # Add trailing slash
        assert len(self.url) > 0
        if self.url[-1] != "/":
            self.url += "/"

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

    def translate(self, q, source="en", target="de"):
        """Translate string

        Args:
            q (str): The text to translate
            source (str): The source language code (ISO 639)
            target (str): The target language code (ISO 639)

        Returns: The translated text
        """

        url = self.url + "translate"

        params = {"q": q, "source": source, "target": target}

        if self.api_key is not None:
            params["api_key"] = self.api_key

        url_params = parse.urlencode(params)

        req = request.Request(url, data=url_params.encode())

        response = request.urlopen(req)

        response_str = response.read().decode()

        return json.loads(response_str)["translatedText"]


if __name__ == '__main__':
    T = Translate()
    T.topic_translation("/Users/kangchieh/Downloads/Bachelorarbeit/wiki_concept/concept/concept.csv")

    #t.topic_translation("/Users/kangchieh/Downloads/Bachelorarbeit/wiki_concept/filter/filter_v1.csv")
    #L = LibreTranslateAPI()
    # L.

    # df = pd.read_csv("/Users/kangchieh/Downloads/Bachelorarbeit/wiki_concept/concept/concept_v1.csv", index_col=0)
    # a = list()
    # b = dict()
    # ft = fasttext.load_model('/Users/kangchieh/Downloads/Bachelorarbeit/cc.de.100.bin')
    #
    # for i in range(len(df)):
    #     DC, EC = df.at[i, 'DC'], df.at[i, 'EC']
    #     if DC not in b:
    #         translated_dc = L.translate(DC)
    #         b[DC] = translated_dc
    #     else:
    #         translated_dc = b[DC]
    #     if EC not in b:
    #         translated_ec = L.translate(EC)
    #         b[EC] = translated_ec
    #     else:
    #         translated_ec = b[EC]
    #     dc_word = ft.get_sentence_vector(translated_dc)
    #     ec_word = ft.get_sentence_vector(translated_ec)
    #     if 1 - spatial.distance.cosine(dc_word, ec_word) >= 0.3:
    #         a.append((translated_dc, translated_ec))
    #     print(i)
    # print(len(a)/len(df))


    # with open("/Users/kangchieh/Downloads/Bachelorarbeit/wiki_concept/concept_de/concept_wiki_de.pkt", 'rb') as f:
    #     extracted_sentences = pickle.load(f)
    # print(extracted_sentences)
    # extracted_sentences = [sentence for key, values in extracted_sentences for sentence in values if sentence in a]
    #
    # with open("/Users/kangchieh/Downloads/Bachelorarbeit/wiki_concept/distributional.pkt", 'rb') as f:
    #     pickle.dump(extracted_sentences, f)








    # t = LibreTranslateAPI(api_key="0b7c26f6-49ab-428e-b593-4085c1a00ec7")
    # dc_list = pd.read_csv("/Users/kangchieh/Downloads/Bachelorarbeit/wiki_concept/topic.csv", index_col=0)
    # index = dc_list.index.values
    # de = []
    #
    # for i in index:
    #     print(i)
    #     translate = t.translate(i)
    #     de.append(translate)
    # print(de)


    # lt = LibreTranslateAPI("https://translate.astian.org/")
    # print(lt.detect("Hello World"))
    # print(lt.translate(lines, "en", "de"))













    # for i in range(1, 20586):
    #     with open("/Users/kangchieh/Downloads/Bachelorarbeit/corpus/test_%s.txt" % i, "r",
    #               encoding="utf-8") as f:
    #         lines = f.readlines()
    #         if len(lines) > 5000:
    #             print(i)

#
#     # Load an En-De Transformer model trained on WMT'19 data:
#     en2de = torch.hub.load('pytorch/fairseq', 'transformer.wmt19.en-de.single_model', tokenizer='moses', bpe='fastbpe', force_reload=True)
#
#     # Access the underlying TransformerModel
#     assert isinstance(en2de.models[0], torch.nn.Module)
#
#     # Translate from En-De
#     de = en2de.translate(
#         'PyTorch Hub is a pre-trained model repository designed to facilitate research reproducibility.')
#     assert de == 'PyTorch Hub ist ein vorgefertigtes Modell-Repository, das die Reproduzierbarkeit der Forschung erleichtern soll.'


    # print(sent_segmentation(text))

