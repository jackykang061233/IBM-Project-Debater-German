import pickle
from ast import literal_eval

import pandas as pd
import re
from collections import Counter
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']


class Meeting_Analysis:
    def __init__(self):
        self.spicy_rank = {"完全不吃辣": 0, "小辣": 1, "中辣": 2, "大辣": 3}
        self.file = pd.read_csv("/Users/kangchieh/Downloads/12/12法蘭克福迎新聚餐.csv", index_col=1)
        self.common_dishes = None
        self.count_sort = None
        self.dish = {'15': 6.8, '24': 28.8, '20': 15.8, '1': 13.8,  '50': 16.8, '56': 11.8, '57': 11.8,
                     '58': 9.8, '54': 11.8, '52': 14.8, '48': 17.8, '43': 14.8, '55': 12.8, '37': 18.8,
                     '19': 7.8, '22': 12.8, '14': 4.8, '28': 16.8, '34': 14.8}

    def spicy(self, x):
        return self.spicy_rank[x]

    def convert(self, x):
        number = re.sub("[^0-9]", " ", x)
        choices = number.split()
        return choices

    def preprocessing(self, file):
        file[file.columns.values[-1]] = file[file.columns.values[-1]].apply(lambda x: self.convert(x))

        out = file.iloc[:, -1].sum()  # concatenate all column values
        count = Counter(out)  # count votes
        self.count_sort = dict(sorted(count.items(), key=lambda item: item[1], reverse=True))  # sort
        print(len(self.count_sort.keys()))

        file['選菜色'] = file.index.to_series().apply(lambda x: self.final_result(x))
        self.common_dishes = file.sort_values(by='選菜色').iloc[:, [2, 4]]

    def final_result(self, x, result=None):
        if result is None:
            result = ['24', '15', '54', '20', '57', '58', '50', '48', '52']
        choice = self.file.loc[x, self.file.columns.values[-1]]
        c = [key for key in result if key in choice]

        return len(c)/len(choice)*len(c) if len(choice) > 0 else 0

    def plot(self, x, y, xlabel, ylabel):
        plt.bar(x, y, width=0.5, color='b')
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.show()

    def table(self):
        # table_1 = ['李宏韋', '廖祐霆', 'Eashen Cheng', '顏維儀', '趙恩宏', 'Jessica Wang', '張凱晶', 'YC Huang', '黃芊融']
        # table_1_big = ['李宏韋', '廖祐霆', 'Eashen Cheng', '顏維儀', '趙恩宏', 'Jessica Wang', '張凱晶', 'YC Huang', '黃蘭茵', '黃芊融']
        # table_2 = [person for person in self.file.index.values if person not in table_1]
        # table_2_big = [person for person in self.file.index.values if person not in table_1_big]
        # return [table_1, table_2, table_1_big, table_2_big]
        table_1 = ['李宏韋', '廖祐霆', '顏維儀', '趙恩宏', 'Jessica Wang', 'YC Huang', '黃蘭茵', '黃芊融']
        table_2 = ['意樸', '謝琳伊', '沈宛靜', 'Eleanore', '邱子瑜', '葉芷瑄', '蕭聖峰']
        # table_3 = ['Eashen Cheng', 'Allen', '郭恩佳', '張凱傑', 'Sylvia ', '張凱晶']
        table_3 = [person for person in self.file.index.values if person not in table_1 and person not in table_2]
        return [table_1, table_2, table_3]


    def process(self):
        for i in range(3):
            self.preprocessing(self.file.loc[self.table()[i], :])
            print(self.file.sort_values(by='可接受辣度', key=lambda x: x.apply(self.spicy)).iloc[:, 2])
            print("\n總人數:", len(self.file)+3)

            # Plot
            #self.plot(self.common_dishes.index.values, self.common_dishes.iloc[:, -1].values, '姓名', '加權分數')
            self.plot(self.count_sort.keys(), self.count_sort.values(), '菜色', '票數')

        # Plot
        # self.plot(self.common_dishes.index.values, self.common_dishes.iloc[:, -1].values, '姓名', '加權分數')
        self.plot(self.count_sort.keys(), self.count_sort.values(), '菜色', '票數')


class get_embedding:
    def __init__(self):
        self.lemma_label = None
        with open('/Users/kangchieh/Downloads/Bachelorarbeit/wiki_concept/expansion_label_de_lemmas.txt', 'rb') as f:
            self.lemma_label = pickle.load(f)
        self.values = list(set(self.lemma_label.values()))
        print(len(self.values))
        self.hidden_states = torch.rand(60000, 4, 1, 3, 768)

    def max_embedding(self, layers_embedding):
        result, _ = torch.max(layers_embedding, dim=0)

        return result

    def min_embedding(self, layers_embedding):
        result, _ = torch.min(layers_embedding, dim=0)

        return result

    def average_embedding(self, layers_embedding):
        result = torch.mean(layers_embedding, dim=0)

        return result

    def find_words(self, tokenized_sentences):
        embedding = {}
        for word in self.values:
            layers = []
            for index_sentence, tokenized_sentence in enumerate(tokenized_sentences):
                for index_token, token in enumerate(tokenized_sentence):
                    if token == word:
                        layers.append(torch.concat([self.hidden_states[index_sentence][i][0][index_token] for i in range(4)], dim=0))
            if layers:
                print(word)
                layers_embedding = torch.stack([layer for layer in layers], dim=0)
                print(layers_embedding.size())

                average = self.average_embedding(layers_embedding)
                print(len(average))
                max = self.max_embedding(layers_embedding)
                print(len(max))
                min = self.min_embedding(layers_embedding)
                print(len(min))

                embedding[word] = [average, max, min]


if __name__ == '__main__':
    from os import listdir
    from os.path import isfile, join
    mypath = "/Users/kangchieh/Downloads/Bachelorarbeit/wiki_concept/wiki_titles"
    onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]
    count = 0
    for f in onlyfiles:
        with open(join(mypath, f), 'rb') as file:
            text = file.read().strip()
            len_chars = len(text)
        count += len_chars
    print(count)
    # g.find_words([['Internet', 'Kaiser', 'Chair'], ['Eating', 'ffod', 'fwfe']])
    # with open("/Users/kangchieh/Downloads/embedding_60000.txt", 'rb') as f:
    #     embedding_60000 = pickle.load(f)
    # print(len(embedding_60000.keys()))
    # import pickle
    # import torch
    #
    #
    # with open("/Users/kangchieh/Downloads/Bachelorarbeit/embedding/embedding_c0.txt", 'rb') as f:
    #     embedding = pickle.load(f)
    # with open("/Users/kangchieh/Downloads/Bachelorarbeit/wiki_concept/de_topic_tokenization.txt", 'rb') as f:
    #      tokenization = pickle.load(f)
    # with open("/Users/kangchieh/Downloads/Bachelorarbeit/wiki_concept/expansion_label_de_lemmas.txt", 'rb') as f:
    #      lemma = pickle.load(f)
    # print(tokenization)
    # print(lemma)
    # input()
    # count = 0
    # for topic, tokens in tokenization.items():
    #     # print(topic)
    #     try:
    #         # print(tokens)
    #         embedding = torch.mean(torch.stack([embedding[lemma[token]][0] for token in tokens], dim=0), dim=0)
    #     except KeyError as e:
    #         count += 1
    #         print('ERROR WORD: ', e)
    # print(count / len(tokenization))
        # for token in doc:
        #     print(token.text, token.pos_, token.dep_)
        #
    # with open("/Users/kangchieh/Downloads/Bachelorarbeit/wiki_concept/expansion_label_de.txt", 'rb') as f:
    #     label_lemma = pickle.load(f)
    # print(label_lemma.keys())

    # analysis = Meeting_Analysis()
    # analysis.table()
    #
    # analysis.process()
    #


