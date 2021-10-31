import pandas as pd
pd.set_option('display.max_columns', 10)
import spacy
import stanza
from stanza.server import CoreNLPClient

class Stanza_Pattern:
    #'([lemma: sein]?) /eine/ /form/ /des|der|von|vom/ '
    #' /,?/ /oder/ /andere/ /Arten/ /des|der|von|vom/'
    def __init__(self):
        self.client = CoreNLPClient(properties='german', annotators=['tokenize', 'ssplit', 'pos', 'lemma', 'ner', 'parse'], timeout=30000, memory='16G')
        self.pattern = ['/ist/ /ein|eine/', '/ist/ /ein/ /Beispiel/ /für/', '/ist/ /eine/ /Form/ /des|der|von|vom/', '/oder/ /andere/ /Arten/ /des|der|von|vom/', '/oder/ /eine/ /andere/ /Art/ /des|der|von|vom/']
        self.nlp_spacy = spacy.load("de_core_news_sm")
        self.nlp_stanza = stanza.Pipeline(lang='de', processors='tokenize, mwt, lemma, pos, depparse')

    def sent_segmentation(self, text):
        sentences = []
        doc = self.nlp_spacy(text)
        assert doc.has_annotation("SENT_START")
        for sent in doc.sents:
            sentences.append(sent.text)
        return sentences

    def tokenization(self, sentence):
        doc = self.nlp_spacy(sentence)

        return [token.text for token in doc]

    def dependency_parser_text(self, text):
        doc = self.nlp_spacy(text)

    def dependency_parser_sentence(self, sentence):
        doc = self.nlp_stanza(sentence)

        return doc.sentences
        # root = [(word.id, word.text) for sent in doc.sentences for word in sent.words if word.head == 0]
        # child = [(word.id, word.text) for sent in doc.sentences for word in sent.words if word.head == root[0][0]]
        # von_root = [(word.head) for sent in doc.sentences for word in sent.words if word.text == 'für']
        #
        # print(child)
        # print(child_root)

    def pattern_extraction_text(self, text, dc, pattern):
        """
        This method consists of several parts:
        1. Do the sentence segmentation of the text
        2. Add the dc to the regex pattern
        3. Extract the pattern from each sentence
        4. Loop through all the matches and get the EC and the whole matched sentences

        Parameters
        ----------
        text: String
            a string of text
        dc: String
            debate concept
        pattern: String
            to match pattern

        Returns
        ------
        List:
            a list contains of every pair of matched (dc, ec, whole sentence)

        """
        sentences = self.sent_segmentation(text)  # sentence segmentation
        pattern_dc = self.pattern_dc_construction(dc, pattern)  # concatenate dc and pattern

        list_rows = []  # store the every pair of (dc, ec, whole sentence)
        for sent in sentences:
            match = self.pattern_extraction_sentence(sent, pattern_dc)  # find the match from the pattern
            dependency_parse = self.dependency_parser_sentence(sent)  # the dependency parser of the sentence

            for index in range(match["sentences"][0]["length"]):
                begin = match['sentences'][0][str(index)]["begin"]  # get the beginning index of each matched sentence
                end = match['sentences'][0][str(index)]["end"]  # get the end index of each matched sentence

                head = dependency_parse[0].words[end-1].head  # the head of the end index word is our potential EC
                ec = " ".join([dependency_parse[0].words[i].text for i in range(end, head)])  # get EC

                whole_sentence = [dependency_parse[0].words[i].text for i in range(begin, head)]  # get our whole matched sentence
                whole_sentence = " ".join(whole_sentence)

                dict_row = {'DC': dc, 'EC': ec, 'Whole Sentence': whole_sentence}  # put in the dict
                list_rows.append(dict_row)
                print(dependency_parse)

        return list_rows

    def pattern_extraction_sentence(self, sentence, pattern):
        matches = self.client.tokensregex(sentence, pattern)
        return matches

    def pattern_dc_construction(self, dc, pattern):
        tokens = self.tokenization(dc)
        pattern_dc = "/ /".join(tokens)  # add the dc to the regex
        pattern_dc = '/' + tokens[0].lower() + '|' + pattern_dc + '/ ' + pattern  # dc can be the first word

        return pattern_dc

    def process(self, text):
        # text = text.lower()
        list_rows = []
        for pattern in self.pattern:
            list_rows += self.pattern_extraction_text(text, 'Das', pattern)
        extracted_sentences = pd.DataFrame(list_rows)
        print(extracted_sentences)


if __name__ == '__main__':
    text = open("/Users/kangchieh/Downloads/Bachelorarbeit/wiki_concept/wiki_titles_de/" + "Waffenkontrolle (Recht)" + ".txt").read()
    database = [
        "Der Freiwilligendienst ist eine Form von freiwilligen Aktivitäten und weist folgende zusätzliche Merkmale auf",
        "Eidetic IntuitionenIntuition soll eine Form des direkten Zugriffs sein.",
        "Vortäuschen ist eine Form der Lüge.",
        "Betrachtet diese Methode als eine Form des spirituellen Hausputzes.",
        "Der Streit ist ja auch eine Form von Begegnung.",
        "Das ist im Grunde genommen eine Form der Prokrastination."]
    database = "Der Freiwilligendienst ist eine Form von freiwilligen Aktivitäten und weist folgende zusätzliche " \
               "Merkmale auf, Eidetic IntuitionenIntuition soll eine Form des direkten Zugriffs sein. Das ist eine " \
               "Form der Lüge. Betrachtet diese Methode als eine Form des spirituellen Hausputzes. Der Streit ist ja " \
               "auch eine Form von Begegnung. Es ist cool. Das ist im Grunde genommen eine Form der Prokrastination. " \
               "Das ist ein Beispiel für einen Ausstrahlungskörpers als Künstler. Das ist ein Beispiel für den " \
               "Beitrag der Kommission zur Friedensbildung und Versöhnung, und dazu ist sie zu beglückwünschen. " \
               "Dies gilt auch für alle unbekannten oder neu entstehenden Viren oder andere Arten von Infektionen." \
               "Gemischte Gesellschaften oder andere Arten von Joint Ventures mit Drittländern sollten sich auf echte, für beide Seiten vorteilhafte Kooperationsvorhaben stützen." \
               "Das oder andere Arten der Ausstattung mit Regalen." \
               "Dieser Eintrag kann für ein Land, eine Organisation oder eine andere Art von Gruppierung stehen." \
               "Wenn Sie ein Dokument oder eine andere Art von Datei an einen Drucker senden, entsteht daraus ein Druckauftrag." \
               "Die Router verfügen über das oder eine andere Art von starke Internetverbindung mit dem Internet."

    S = Stanza_Pattern()
    # print(S.tokenization('Der Freiwilligendienst'))
    S.process(database)
    # S.pattern_extraction_text(database)
    # print(S.pattern_extraction_sentence(database))




# X ist eine Art [von] Y
# X ist eine Form von Y
# X ist ein Beispiel für Y
# X ist ein Spezialfall von Y
#
# X oder andere Arten von Y
# X oder eine andere Art von Y
# X oder andere[r/s] Y
# X und andere[r/s] Y
# X und andere Arten von Y