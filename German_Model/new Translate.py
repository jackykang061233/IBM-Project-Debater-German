import json
import pickle
from os import listdir
from os.path import isfile, join
from urllib import parse, request
from urllib.error import HTTPError

import pandas as pd
import requests
import spacy


class LibreTranslateAPI:
    def __init__(self):
        self.nlp_spacy = spacy.load("en_core_web_sm")
        self.headers = {
        'accept': 'application/json',
        'Content-Type': 'application/x-www-form-urlencoded',}
        self.data = {
        'source': 'en',
        'target': 'de',
        'format': 'text',
        'api_key': '0b7c26f6-49ab-428e-b593-4085c1a00ec7'}

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

    def translate(self, q):
        self.data['q'] = q
        response = requests.post('https://translate.argosopentech.com/translate', headers=self.headers, data=self.data)
        try:
            translate = response.json()['translatedText']
        except KeyError:
            print(1)
        return translate

    def process(self):
        path = "/Users/kangchieh/Downloads/Bachelorarbeit/corpus_de/"
        files = [f for f in listdir(path) if isfile(join(path, f))]
        not_exist = []
        for i in range(1, 20586):
            try:
                text = open("/Users/kangchieh/Downloads/Bachelorarbeit/corpus_de/test_%s.txt" % i, "r",
                            encoding="utf-8").read()
            except:
                not_exist.append(i)
        a = [n for n in not_exist if n > 4133]
        b = [n for n in a if n < 7000]
        # print(len(a))

        for i in not_exist:
            print(i)
            sentence_length = 0
            sentence = ''
            translate = ''
            try:
                with open("/Users/kangchieh/Downloads/Bachelorarbeit/corpus/test_%s.txt" % i, 'r') as f:
                    lines = f.readlines()

                sentences = self.sent_segmentation(lines[0])
                for sent in sentences:
                    sentence_length += len(sent)
                    sentence += sent
                    if sentence_length > 4500:
                        translate += self.translate(sentence)
                        sentence_length = 0
                        sentence = ''

                translate += self.translate(sentence)

                with open("/Users/kangchieh/Downloads/Bachelorarbeit/corpus_de/test_%s.txt" % i, "w",
                          encoding="utf-8") as f:
                    f.write(translate)

            except HTTPError:
                print('Error: %s' % i)
                with open("/Users/kangchieh/Downloads/Bachelorarbeit/exception_10.txt", "a",
                          encoding="utf-8") as exception:
                    exception.write(str(i) + '\n')
            except TypeError:
                print('Error: %s' % i)
                with open("/Users/kangchieh/Downloads/Bachelorarbeit/exception_10.txt", "a",
                          encoding="utf-8") as exception:
                    exception.write(str(i) + '\n')

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

if __name__ == '__main__':
    L = LibreTranslateAPI()
    df_en = pd.read_csv("/Users/kangchieh/Downloads/Bachelorarbeit/wiki_concept/expansion_label_en.csv",
                        index_col=0)
    topics = list(set(list(df_en.DC.values)+list(df_en.EC.values)))
    translated = {}

    for t in topics:
        print(t)
        translated[t] = L.translate(t)

    with open('/Users/kangchieh/Downloads/Bachelorarbeit/wiki_concept/translated/topic_translation_LiberTranslate.txt',
              'wb') as f:
        pickle.dump(translated, f)
