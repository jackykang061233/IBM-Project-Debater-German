# regular expression
import re
# wikipedia api
import wikipediaapi
# postag and chunk
import spacy
# basic functions
import pandas as pd
import pickle


class Pattern:
    """
    This class is aimed to find the pattern of the potential extraction candidates and extract the EC from the matched
    sentences.
    ...

    Attributes
    ----------
    spacy: Object
        a small English Spacy pipeline trained on written web text that containing tok2vec, tagger, parser, senter,
        attribute_ruler, lemmatizer, ner
    dc : Dataframe
        a Dataframe containing our Debate Topics with wiki outlinks
    consistent_pattern : List
        a list containing the matching consistent patterns (re)
    Methods
    -------
    extract_concept_wiki():
        uses the matching patterns to find the potential EC (normally a sentence here)
    get_EC(data):
        uses another method chunk to find all the matchings found by self.extract_concept_wiki and lists them in the dataframe with
        3 parts: DC (Debate concept also our main topic), EC (Expansion concept also concept in the matched sentence)
        and Original (the original sentence)
    find_synonym():
        finds the pattern '[our title], also known as [synonym]' in wiki articles,
        cause in wikipedia a title can have synonyms
    re_match(pattern, text):
        uses re to find the matched sentence in our text
    postag(sentence):
        finds the postag of a sentence
    chunk(sentence):
        finds the first noun chunk of a sentence and if 'of' is between the first noun chunk and the second connect them
    """

    def __init__(self):
        self.spacy = spacy.load("en_core_web_sm")  # spacy postag
        self.dc = pd.read_csv('/Users/kangchieh/Downloads/Bachelorarbeit/wiki_concept/topic_with_outlinks.csv',
                              converters={'outlinks': eval}, index_col=0)
        self.consistent_pattern = [r' is a)\b((\s*\w+\-*\'*){1,10})',
                                   r' is an)\b((\s*\w+\-*\'*){1,10})',
                                   r' is a kind of)\b((\s*\w+\-*\'*){1,10})',
                                   r' is a form of)\b((\s*\w+\-*\'*){1,10})',
                                   r' is an example of)\b((\s*\w+\-*\'*){1,10})',
                                   r' is a special case of)\b((\s*\w+\-*\'*){1,10})',
                                   r' or other)\b((\s*\w+\-*\'*){1,10})',
                                   r' or other types of)\b((\s*\w+\-*\'*){1,10})',
                                   r' or other kinds of)\b((\s*\w+\-*\'*){1,10})',
                                   r' or another type of)\b((\s*\w+\-*\'*){1,10})',
                                   r' and other)\b((\s*\w+\-*\'*){1,10})',
                                   r' and other types of)\b((\s*\w+\-*\'*){1,10})',
                                   r' and other kinds of)\b((\s*\w+\-*\'*){1,10})']
        # r'((\w+\-*\s*){1,10})\bsuch as ',
        # r'((\w+\-*\s*){1,10})\b(including ',
        # r'((\w+\-*\s*){1,10})\b(e\.g\. ']

    def extract_concept_wiki(self):
        """
        This function uses the matching patterns to find the potential EC (normally a sentence here)
        """
        topic = {concept: [] for concept in self.dc.index.values}  # get our main topics and create an empty list
        synomyms = self.find_synonym()  # get synonyms of our main topics
        for concept in self.dc.index.values:
            print(concept)
            outlinks = self.dc.loc[concept, "outlinks"]  # get all the outlinks of main topic
            outlinks.append(concept)  # add also main topic itself
            for outlink in outlinks:
                try:
                    text = open(
                        "/Users/kangchieh/Downloads/Bachelorarbeit/wiki_concept/wiki_titles/" + outlink + ".txt",
                        "r").read()
                    for pattern in self.consistent_pattern:
                        concept_pattern = r'(' + concept.lower() + pattern  # create pattern from main topic
                        match = self.re_match(concept_pattern, text.lower())  # find re match in outlink files
                        if match:
                            topic[concept].append(match[0][1])  # add the sentence matched with the pattern

                        if concept in synomyms:  # if a concept has a synonym then also find the match of synonyms
                            for synomym in synomyms[concept]:
                                synomym_pattern = r'(' + synomym.lower() + pattern
                                match = self.re_match(synomym_pattern, text.lower())
                                if match:
                                    topic[concept].append(match[0][1])
                except FileNotFoundError:  # if outlink file not exists
                    pass
            topic[concept] = list(set(topic[concept]))  # delete duplicates

        with open("/Users/kangchieh/Downloads/Bachelorarbeit/wiki_concept/concept_wiki_con_v3.pkt", "wb") as f:
            pickle.dump(topic, f)

    def get_EC(self, data):
        """
        This function uses another method chunk to find all the matchings found by self.extract_concept_wiki and lists them in the dataframe with
        3 parts: DC (Debate concept also our main topic), EC (Expansion concept also concept in the matched sentence)
        and Original (the original sentence)
        """
        df = pd.DataFrame(columns=['DC', 'EC', "Original"])
        last_ec = None
        for key, value in data.items():
            if value:  # if there is at least one matching
                value.sort()
                for v in value:
                    ec = self.chunk(v[1:])  # Starts from 1 because of the blank in each of the sentence and we try
                                            # to find the first noun as our EC
                    if ec and last_ec != ec:  # if this EC exists and not added already
                        df.loc[df.shape[0]] = [key, ec, v]  # Added at the bottom of the df
                    last_ec = ec

        df.to_csv("/Users/kangchieh/Downloads/Bachelorarbeit/wiki_concept/concept_wiki_con.csv")

    def find_synonym(self):
        """
        In wikipedia a title can have synonyms. This function finds the pattern '[our title], also known as [synonym]'
        """
        synonym = {}
        for concept in self.dc.index.values:
            text = open("/Users/kangchieh/Downloads/Bachelorarbeit/wiki_concept/wiki_titles/" + concept + ".txt",
                        "r").read()
            pattern = concept.lower() + r'(, also known as)\b((\s*\w+\-*)*)'  # our pattern also known as
            match = self.re_match(pattern, text.lower())
            if match:
                synonym[concept] = []
                if ' or ' in match[0][1]:  # if a concept has more than two synonyms
                    text = match[0][1].split(' or ')  # split two synonyms
                    for t in text:
                        if t[0] == ' ':
                            synonym[concept].append(t[1:])
                        else:
                            synonym[concept].append(t)
                else:  # if a concept has only one synonym
                    synonym[concept].append(match[0][1][1:])
        synonym['Doping in sport'] = ['doping']  # The synonym of 'Doping in sport' is 'doping
        return synonym

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

    def chunk(self, sentence):
        """
        This function finds the first noun chunk of a sentence and
        if 'of' is between the first noun chunk and the second connect them
        Parameters
        ----------
        sentence: String
            a string of sentence
        Returns
        ------
        List:
            the first noun chunk of the input sentence or the first noun chunk + 'of' + the second noun chunk
            if 'of' is between the first noun chunk and the second noun chunk
        """
        chunk = []
        sentence_chunk = self.spacy(sentence)
        for word in sentence_chunk.noun_chunks:  # find only the noun chunk
            chunk.append(word.text)
        if len(chunk) > 1:  # in case that the chunk consists of more than 1 word
            if chunk[0] + ' of ' + chunk[1] in sentence:  # if of appears then combine the first and the second chunk
                return chunk[0] + ' of ' + chunk[1]
        try:
            return chunk[0]
        except IndexError:  # if not exists
            return []


if __name__ == "__main__":
    p = Pattern()
    print(p.postag('hello youtube'))
    # p.extract_concept_wiki()

    with open("/Users/kangchieh/Downloads/Bachelorarbeit/wiki_concept/concept_wiki_con_v2.pkt", "rb") as f:
        unserialized_data = pickle.load(f)
    print(unserialized_data)
    # p.get_EC(unserialized_data)
