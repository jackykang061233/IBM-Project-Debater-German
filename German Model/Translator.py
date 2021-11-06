from deep_translator import LingueeTranslator
import wikipediaapi
import spacy

# class Translator:
#     def __init__(self):
#         pass
#
#     def

if __name__ == '__main__':
    def sent_segmentation(text):
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
        nlp_spacy = spacy.load("de_core_news_sm")
        doc = nlp_spacy(text)
        assert doc.has_annotation("SENT_START")
        for sent in doc.sents:
            sentences.append(sent.text)  # append sentence to list
        return sentences
    wiki_wiki = wikipediaapi.Wikipedia('en')

    page_py = wiki_wiki.page('Gun Control')
    # page_py_de = page_py.langlinks['de']
    text = page_py.text
    print(text)
    possible_list = ['See Also', 'Notes' 'References' 'Bibliography' 'Further reading', 'External links']

    # print(sent_segmentation(text))

