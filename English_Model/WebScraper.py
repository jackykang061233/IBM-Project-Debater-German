# basic functions
import pandas as pd
# wikipedia api
import wikipediaapi



class wikiscraper:
    """
    A class used to web scrap the wiki pages related to our debate topics

    ...

    Attributes
    ----------
    dc : Dataframe
        a Dataframe containing our Debate Topics
    dc_outlink : Dataframe
        a Dataframe containing our Debate Topics and their wiki outlinks
    wiki : Object
        a Wikipedia object with the language parameter
    Methods
    -------
    get_outlinks:
        gets the outlinks of each wiki topic
    get_articles:
        gets the wiki page of a wiki topic
    get_all_articles:
        get all the wiki pages from all the outlinks of each main topic
    """

    def __init__(self):
        self.dc = pd.read_csv('wiki_concept/topic.csv',
                              index_col=0)  # our main topics
        # Because that we get our column "outlinks" as a string, we use converters={'outlinks': eval} to convert
        # string to list
        self.dc_outlink = pd.read_csv('wiki_concept/topic_with_outlinks.csv',
                                      converters={'outlinks': eval}, index_col=0)
        self.wiki = wikipediaapi.Wikipedia('de')

    def get_outlinks(self):
        """ This function gets the outlinks of each wiki topic"""
        outlinks = []
        for dc in self.dc.index.values:
            page = self.wiki.page(dc)
            outlinks.append(list(page.links.keys()))
        self.dc['outlinks'] = outlinks
        self.dc.to_csv('wiki_concept/topic_with_outlinks.csv')

    def get_articles(self, title):
        """ This function gets the wiki page of a wiki topic"""
        page = self.wiki.page(title)
        text = page.text
        print(text)
        try:
            with open("wiki_concept/wiki_titles_de/" + title + ".txt", "w",
                      encoding="utf-8") as f:
                f.write(text)
        except FileNotFoundError:  # in case that / occurs in the title
            with open("exception.txt", "a") as ex:
                ex.write(title + "\n")
                ex.close()

    def get_all_articles(self):
        """ This function get all the articles from all the outlinks of each main topic"""
        flag = False
        for concept in self.dc_outlink.index.values:
            # if flag:
            #     print(concept)
            self.get_articles(concept)
            outlinks = self.dc_outlink.loc[concept, "outlinks"]
            for outlink in outlinks:
                # if outlink[:5] == "Portal":
                #     continue
                # if concept == 'Internet' and outlink == "List of countries by number of Internet users":
                #     flag = True
                if flag:
                    print(outlink)
                    self.get_articles(outlink)


if __name__ == "__main__":
    w = wikiscraper()
    w.get_articles("Waffenkontrolle (Recht)")


