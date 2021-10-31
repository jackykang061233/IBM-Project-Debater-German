import stanza
from stanza.server import CoreNLPClient


def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press ⌘F8 to toggle the breakpoint.


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    text = "Acai-Beere ist eine Art von Obst gewöhnlich in tropischen Regenwäldern Zentral-und Südamerika ."
    nlp_stanza = stanza.Pipeline(lang='de', processors='tokenize, mwt, lemma, pos, depparse')
    doc = nlp_stanza(text)
    print(doc.sentences)
    whole_sentences = []
    pattern = ' ist ein '
    punc = '''!()-[]{};:'"\,<>./?@#$%^&*_~'''

    # client = CoreNLPClient(properties='german', annotators=['tokenize', 'ssplit', 'pos', 'lemma', 'ner', 'parse'],
    #                    timeout=30000,
    #                    memory='16G')
    #     # submit the request to the server
    # ann = client.annotate(text)
    #     #
    #     # # get the first sentence
    # sentence = ann.sentence[0]
    # print(sentence)
    #
    #     #
    #     # # get the constituency parse of the first sentence
    #     # print('---')
    #     # print('constituency parse of first sentence')
    #     # constituency_parse = sentence.parseTree
    #     # print(constituency_parse)
    #     #
    #     # # get the first subtree of the constituency parse
    #     # print('---')
    #     # print('first subtree of constituency parse')
    #     # print(constituency_parse.child[0])
    # pattern = '/ist/ /ein/ /Beispiel/ /für/ []{0,5} ([pos: NOUN])'
    # matches = client.tokensregex(text, pattern)
    # # sentences contains a list with matches for each sentence.
    # print(matches["sentences"][0]["0"]["text"])

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
