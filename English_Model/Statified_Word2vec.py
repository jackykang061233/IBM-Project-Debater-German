# Pickle
import pickle
# Numpy
import numpy as np


class Statified_Word2vec:
    def __init__(self):
        self.lemma_de = self.get_lemma()
        self.embedding = pickle.load(open("/Users/kangchieh/Downloads/Bachelorarbeit/statified word embedding/german_embedding.txt", "rb"))

    def get_lemma(self):
        with open("/Users/kangchieh/Downloads/Bachelorarbeit/wiki_concept/expansion_label_de_lemmas.txt", 'rb') as f:
            lemma = pickle.load(f)
        return lemma

    def fasttext_embedding(self, words):
        initial_vec = np.zeros(768)
        count = 0
        for word in words:
            word_embedd = self.embedding[word]
            norm = self.l2_norm(word_embedd)
            if norm > 0:
                initial_vec = np.add(initial_vec, word_embedd*(1/norm))
                count += 1
        return initial_vec/count

    def l2_norm(self, vec):
        return np.linalg.norm(vec)

    def max_embedding(self, words):
        vec = np.ones(768)*(-np.inf)
        for word in words:
            word_embedd = self.embedding[word]
            if np.any(word_embedd):
                vec = np.maximum(vec, word_embedd)
        return vec

    def min_embedding(self, words):
        vec = np.ones(768) * (np.inf)
        for word in words:
            word_embedd = self.embedding[word]
            if np.any(word_embedd):
                vec = np.minimum(vec, word_embedd)
        return vec

    def average_embedding(self, words):
        word_embedd = [self.embedding[word] for word in words]
        return np.mean(word_embedd, axis=0)

    # def update_embedding(self):
    #     new_embedding = {}
    #     count = 1
    #     for lemma in self.lemma_de.values():
    #         try:
    #             max = [embedd[lemma][1] for embedd in self.embedding_dicts]
    #             max, _ = torch.max(torch.stack([max_embedd for max_embedd in max], dim=0), dim=0)
    #
    #             min = [embedd[lemma][2] for embedd in self.embedding]
    #             min, _ = torch.min(torch.stack([min_embedd for min_embedd in min], dim=0), dim=0)
    #
    #             new_embedding[lemma] = [max, min]
    #         except KeyError:
    #             count += 1
    #     with open("/Users/kangchieh/Downloads/Bachelorarbeit/embedding/final.txt", 'wb') as f:
    #         pickle.dump(new_embedding, f)
    def update_min_embedding(self):
        pass

    def update_average_embedding(self):
        pass


if __name__ == "__main__":
    s = Statified_Word2vec()
    words = ['schlecht', 'Todesstrafe']
    print(s.average_embedding(words))
    # a = np.array([1, 2, 3])
    # print(s.l2_norm(a))



