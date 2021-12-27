import pickle
from os import listdir
from os.path import isfile, join

# Pytorch
import torch


class statified_word2vec:
    def __init__(self, path):
        self.path = path
        #self.folders = self.get_embedding_files()
        #self.embedding_dicts = self.get_embedding_dicts()
        self.lemma_de = self.get_lemma()
        self.embedding = self.get_embedding()

    def get_embedding_files(self):
        folders = [f for f in listdir(self.path) if isfile(join(self.path, f))]
        folders.sort(key=lambda x: int(x.partition("embedding_c")[2].partition(".txt")[0]))
        return folders

    def get_embedding_dicts(self):
        embedding_files = []
        for file in self.folders:
            with open(join(self.path, file), 'rb') as f:
                embedding_files.append(pickle.load(f))
        return embedding_files

    def get_embedding(self):
        with open("/Users/kangchieh/Downloads/Bachelorarbeit/embedding/final.txt", 'rb') as f:
            embedding = pickle.load(f)
        return embedding

    def get_lemma(self):
        with open("/Users/kangchieh/Downloads/Bachelorarbeit/wiki_concept/expansion_label_de_lemmas.txt", 'rb') as f:
            lemma = pickle.load(f)
        return lemma

    def max_embedding(self, word):
        return self.embedding[word][0]

    def min_embedding(self, word):
        return self.embedding[word][1]

    def average_embedding(self):
        pass

    def update_embedding(self):
        new_embedding = {}
        count = 1
        for lemma in self.lemma_de.values():
            try:
                max = [embedd[lemma][1] for embedd in self.embedding_dicts]
                max, _ = torch.max(torch.stack([max_embedd for max_embedd in max], dim=0), dim=0)

                min = [embedd[lemma][2] for embedd in self.embedding]
                min, _ = torch.min(torch.stack([min_embedd for min_embedd in min], dim=0), dim=0)

                new_embedding[lemma] = [max, min]
            except KeyError:
                count += 1
        with open("/Users/kangchieh/Downloads/Bachelorarbeit/embedding/final.txt", 'wb') as f:
            pickle.dump(new_embedding, f)
    def update_min_embedding(self):
        pass

    def update_average_embedding(self):
        pass


if __name__ == "__main__":
    s = statified_word2vec("/Users/kangchieh/Downloads/Bachelorarbeit/embedding")
    s.update_embedding()

