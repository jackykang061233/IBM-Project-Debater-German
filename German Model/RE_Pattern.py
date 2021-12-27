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
        self.dc = pd.read_csv('/Users/kangchieh/Downloads/Bachelorarbeit/wiki_concept/topic.csv',
                              converters={'outlinks': eval}, index_col=0)
        # self.consistent_pattern = [r' ist ein|eine)\b((\s*\w+\-*\'*){1,10})',
        #                            r' ist eine Art des|der|von|vom)\b((\s*\w+\-*\'*){1,10})',
        #                            r' ist eine Form des|der|von|vom)\b((\s*\w+\-*\'*){1,10})',
        #                            r' is a form of)\b((\s*\w+\-*\'*){1,10})',
        #                            r' is an example of)\b((\s*\w+\-*\'*){1,10})',
        #                            r' is a special case of)\b((\s*\w+\-*\'*){1,10})',
        #                            r' or other)\b((\s*\w+\-*\'*){1,10})',
        #                            r' or other types of)\b((\s*\w+\-*\'*){1,10})',
        #                            r' or other kinds of)\b((\s*\w+\-*\'*){1,10})',
        #                            r' or another type of)\b((\s*\w+\-*\'*){1,10})',
        #                            r' and other)\b((\s*\w+\-*\'*){1,10})',
        #                            r' and other types of)\b((\s*\w+\-*\'*){1,10})',
        #                            r' and other kinds of)\b((\s*\w+\-*\'*){1,10})']
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

    def extract_concept(self):
        """
        This function uses the matching patterns to find the potential EC (normally a sentence here)
        """
        topic = {concept: [] for concept in self.dc.index.values}  # get our main topics and create an empty list
        # synomyms = self.find_synonym()  # get synonyms of our main topics
        for concept in self.dc.index.values:
            print(concept)
            with open("/Users/kangchieh/Downloads/Bachelorarbeit/match_sentences/de_libre_shorten.txt",
                      "rb") as f:
                extracted_sentences = pickle.load(f)
            for lines in extracted_sentences:
            # for i in range(1, 20586):
            #     try:
            #         with open("/Users/kangchieh/Downloads/Bachelorarbeit/corpus_de/test_%s.txt" % i, 'r') as f:
            #             lines = f.readlines()
            #         lines = ' '.join(lines)
            #
            #     except FileNotFoundError:
            #         continue

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
                                with open("/Users/kangchieh/Downloads/Bachelorarbeit/exception.txt", "a",
                                          encoding="utf-8") as exception:
                                    exception.write(concept + '\n')

            topic[concept] = list(set(topic[concept]))  # delete duplicates

        with open("/Users/kangchieh/Downloads/Bachelorarbeit/wiki_concept/concept_wiki_de_libre_shorten.pkt", "wb") as f:
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

    def extract_dc_sentence(self):
        sentences = []
        for concept in self.dc.index.values:
            print(concept)
            for i in range(1, 20586):
                try:
                    with open("/Users/kangchieh/Downloads/Bachelorarbeit/corpus/test_%s.txt" % i, 'r') as f:
                        text = f.readlines()
                    text = ' '.join(text)
                    c = ' ' + concept + ' '
                    sentences += [sentence + '.' for sentence in text.split('. ') if c.lower() in sentence.lower() and len(sentence) < 1025]

                except FileNotFoundError:
                    continue
        with open("/Users/kangchieh/Downloads/Bachelorarbeit/match_sentences/en_shorten.txt", "wb") as f:
            pickle.dump(sentences, f)


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
    #p.extract_concept()
    with open("/Users/kangchieh/Downloads/Bachelorarbeit/match_sentences/en_shorten.txt", "rb") as f:
        a = pickle.load(f)

    # print(l)

    # text = "Waffenkontrolle ist eine Form von essen und ich mag h'schokolade. Waffenkontrolle ist eine Form der text"
    # pattern = r'(Waffenkontrolle ist eine Form des|der|von|vom)\s+(.+?)\.'
    # print(p.re_match(pattern, text))
    #p.extract_concept()



    # with open("/Users/kangchieh/Downloads/Bachelorarbeit/wiki_concept/concept_wiki_con_v2.pkt", "rb") as f:
    #     unserialized_data = pickle.load(f)
    # print(unserialized_data)
    # p.get_EC(unserialized_data)