# string to list
from ast import literal_eval

from matplotlib import pyplot as plt

from HelpFunctions_de import Wiki, Wordnet, Word2vec, Sentiment_Analysis
# lemmatization
import spacy
# sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_squared_error, precision_score, recall_score
from sklearn.metrics import f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
# basic functions
import pickle
import numpy as np
import pandas as pd


class GetFeature:
    """
        A class used to represent an Animal

        ...

        Attributes
        ----------
        df : Dataframe
            a formatted string to print out what the animal says

        Methods
        -------
        get_wiki(df)
            Prints the animals name and what sound it makes
        get_wordnet(df)
        sentiment(text)
        sentiment_analysis(df)
        load_sentiment_analysis(path, df)
        load_distributional_similarity(path, df)
        word_embedding(df)
        processing()
        """

    def __init__(self):
        """
        Parameters
        ----------
        df : Dataframe
            A Dataframe consists of
            1. DC
            2. EC
            3. stop_words
            4. substring
            5. ner
            6. distributional_similarity
            7. DC_freq
            8. EC_freq
            9. good expansion
            10. DC_embedding
            11. EC_embedding
            12. shared_categories
            13. shared_links
            14. hypernym
            15. hyponym
            16. co-hypernym
            17. synonym
            18. DC_Polarity_diff
            19. EC_Polarity_diff
            20. freq_ratio
        """
        pass

    def get_wiki(self, df, path_cat, path_link):
        w = Wiki(df)
        w.processing(path_cat, path_link)
        with open("/Users/kangchieh/Downloads/Bachelorarbeit/wiki_concept/wiki/cat_de.pkt", "rb") as f:
            cat = pickle.load(f)
        with open("/Users/kangchieh/Downloads/Bachelorarbeit/wiki_concept/wiki/link_de.pkt", "rb") as f:
            link = pickle.load(f)
        df['shared_categories'] = df.apply(lambda row: cat.get((row.DC, row.EC)), axis=1)
        df['shared_links'] = df.apply(lambda row: link.get((row.DC, row.EC)), axis=1)
        return df

    def get_wordnet(self, df):
        w = Wordnet(df)
        return w.processing()

    def get_sentiment(self, df, path):
        s = Sentiment_Analysis(df)
        s.processing(path)
        with open("/Users/kangchieh/Downloads/Bachelorarbeit/wiki_concept/sentiment/sentiment_de.pkt", "rb") as f:
            sentiment = pickle.load(f)
        df['DC_sentiment'] = df["DC"].apply(lambda x: sentiment.get(x))
        df['EC_sentiment'] = df["EC"].apply(lambda x: sentiment.get(x))
        return df

    def word_embedding(self, df):
        w = Word2vec(df)
        word2vec = w.embedding_spacy()
        dc_embedding = []
        ec_embedding = []
        for i in range(len(df)):
            DC, EC = df.at[i, 'DC'], df.at[i, 'EC']
            dc_embedding.append(list(word2vec[DC]))
            ec_embedding.append(list(word2vec[EC]))

        df['DC_embedding'] = dc_embedding
        df['EC_embedding'] = ec_embedding
        return df

    def processing(self, df, sentiment_path=None, cat_path=None, link_path=None):
        # wordembed = self.word_embedding(self.df)
        wiki = self.get_wiki(df, cat_path, link_path)
        wordnet = self.get_wordnet(wiki)
        final_result = self.get_sentiment(wordnet, sentiment_path)
        # final_result = self.load_distributional_similarity(self.df, sentiment)

        final_result['freq_ratio'] = final_result.apply(
            lambda row: min(row.DC_freq / row.EC_freq, row.EC_freq / row.DC_freq) if row.DC_freq != 0 and row.EC_freq != 0 else 1/max(row.DC_freq, row.EC_freq), axis=1)

        return final_result


class Training:
    def __init__(self, df, load_from_path, lang):
        """
        Parameters
        ----------
        df : Dataframe
            A Dataframe consists of
            0. DC
            1. EC
            2. Original
            3. stop_words
            4. substring
            5. ner
            6. distributional_similarity
            7. DC_freq
            8. EC_freq
            9. good expansion
            10. DC_embedding
            11. EC_embedding
            12. shared_categories
            13. shared_links
            14. hypernym
            15. hyponym
            16. co-hypernym
            17. synonym
            18. DC_Polarity_diff
            19. EC_Polarity_diff
            20. freq_ratio
        log_reg: sklearn.linear_model
            a sklearn model which performs binary or multiclass classification
        new_df: Dataframe
            new_df is predefined as None but will later be assigned to an adjusted version of df
        X: Dataframe
            X is predefined as None but will later be assigned to our training data
        Y: Series
            Y is predefined as None but will later be assigned to our output target
        """
        self.df = df
        self.load_from_path = load_from_path
        self.lang = lang
        self.new_df = None
        self.X, self.y = None, None
        self.label = pd.read_csv("/Users/kangchieh/Downloads/Bachelorarbeit/wiki_concept/expansion_label_de.csv", index_col=0)
        self.label_en = pd.read_csv("/Users/kangchieh/Downloads/Bachelorarbeit/wiki_concept/expansion_label_en.csv",
                                 index_col=0)

    def cleaning_data(self, df, drop=None):
        """ This function selects only the features needed in the training"""
        if drop is None:
            drop = []
        X = df.drop(['DC', 'EC', 'stop_words', 'substring',  'DC_freq', 'EC_freq', 'filter out']+drop, axis=1)
        y = X.pop('good expansion')  # target output

        #self.X = self.new_df  # training input

        # Since our lists are saved as string we have to transform them back to list
        if self.load_from_path:
            X['DC_embedding'] = X['DC_embedding'].apply(literal_eval)
            X['EC_embedding'] = X['EC_embedding'].apply(literal_eval)

        return X, y

    def building_pipeline(self, df):
        """ This function scales two columns 'shared_links' and 'shared_categories' with StandardScaler()"""
        pipeline = Pipeline([
            ('std_scaler', StandardScaler()),
        ])
        full_pipeline = ColumnTransformer([
            ("num", pipeline, ['shared_links', 'shared_categories']),
        ])
        df[['shared_links', 'shared_categories']] = pd.DataFrame(full_pipeline.fit_transform(df))
        return df

    def gettint_input(self, df):
        """ This function gets all training data from the same row to a list"""
        # self.X['shared_links'] = StandardScaler().fit_transform(self.X['shared_links'].values).tolist( )
        # self.X['shared_categories'] = StandardScaler().fit_transform(self.X['shared_categories'].values)
        df["input"] = df.apply(lambda row: self.row_to_list(row), axis=1)
        df["input"] = df["input"].apply(lambda x: self.flatten(x))
        return df

    def row_to_list(self, row):
        """
        This function make a Dataframe row values to one list
        Parameters
        ----------
        row: Series
            a row in a Dataframe

        Returns
        -------
        list:
            a from Series converted nested list

        """
        return row.tolist()

    def flatten(self, l):
        """
        This functions flattens the multidimensional object in a list to one dimension

        Parameters
        ----------
        l: List
            a from above function converted nested list

        Returns
        -------
        list:
            a list contains only of one dimensional object

        """
        new_l = []
        for sub in l:
            try:
                for item in sub:
                    new_l.append(item)
            except TypeError:
                new_l.append(sub)
        return new_l

    def split_data(self, x, y, size):
        """
        This functions splits our input and output data into train and test data

        Parameters
        ----------
        x: Numpy array
            a numpy array of input data
        y: Numpy array
            a numpy array of target data
        Returns
        ------
        List:
            a list contains of our split data

        """
        return train_test_split(x, y, test_size=size)

    def logistic_regression(self, X, y, split_ratio=0.2):
        """ This function performs logistic regression and prints the result at the end"""
        input_X = np.array([np.array(x) for x in X["input"].values])
        input_y = np.array(y.values)

        X_train, X_test, y_train, y_test = self.split_data(input_X, input_y, split_ratio)
        log_reg = LogisticRegression(max_iter=5000)
        log_reg.fit(X_train, y_train)

        test_predictions = log_reg.predict(X_test)

        print('Logistic Regression:')
        print("Predictions: ", test_predictions)
        print("Labels:      ", y_test)
        print("\nSqrt MSE:    ", self.mean_square_error(test_predictions, y_test))
        print("f1 Score:    ", self.f1_score(y_test, test_predictions))
        print("Precision:   ", precision_score(y_test, test_predictions, average="weighted"))
        print("Recall:      ", recall_score(y_test, test_predictions, average="weighted"))
        return y_test, test_predictions

    def cross_validation(self, model, X, y):
        """
        This function performs cross validation and prints the result at the end
        Parameters
        ----------
        model: sklearn model
            a model to use to fit the data
        """
        input_X = np.array([np.array(x) for x in X["input"].values])
        input_y = np.array(y.values)
        scores = cross_val_score(model, input_X, input_y,
                                 scoring="neg_mean_squared_error", cv=5)
        tree_rmse_scores = np.sqrt(-scores)
        # print("\nCross Validation:")
        # print("Scores:", tree_rmse_scores)
        # print("Mean:  ", tree_rmse_scores.mean())
        # print("Standard deviation:", tree_rmse_scores.std())

    def mean_square_error(self, y_test, test_predictions):
        reg_mse = mean_squared_error(y_test, test_predictions)
        reg_rmse = np.sqrt(reg_mse)

        return reg_rmse

    def f1_score(self, y_test, test_predictions):
        return f1_score(y_test, test_predictions, average='weighted')

    def f1_score_compare(self, lang='de', drop=None, times=100, path=None, df=None):
        f1_score_1 = []
        if path is not None:
            df = pd.read_csv(path, index_col=0)
        f1_score_2 = []
        for i in range(times):
            y_test_1, test_predictions_1 = self.process(self.df, drop, self.lang)
            y_test_2, test_predictions_2 = self.process(df, drop, lang)
            f1_score_1.append(self.f1_score(y_test_1, test_predictions_1))
            f1_score_2.append(self.f1_score(y_test_2, test_predictions_2))
        print(f1_score_1)
        print(f1_score_2)
        plt.plot(f1_score_1, label='Spacy')
        plt.plot(f1_score_2, label='Fasttext')
        plt.xlabel("Times")
        plt.ylabel("f1 score")
        plt.legend(loc="upper left", fontsize=10)
        plt.show()

    def process(self, df, drop, lang):
        if lang == 'de':
            df = df.merge(self.label, on=['DC', 'EC'])
        elif lang == 'en':
            df = df.merge(self.label_en, on=['DC', 'EC'])

        df.to_csv("/Users/kangchieh/Downloads/Bachelorarbeit/wiki_concept/test.csv")

        X, y = self.cleaning_data(df, drop)
        X = self.building_pipeline(X)
        X = self.gettint_input(X)
        y_test, test_predictions = self.logistic_regression(X, y)
        self.cross_validation(LogisticRegression(), X, y)

        return y_test, test_predictions


if __name__ == "__main__":
    # data = pardata.load_dataset('wikipedia_oriented_relatedness')
    # print(data)


    # path = "/Users/kangchieh/Downloads/Bachelorarbeit/corpus_de/"
    # files = [f for f in listdir(path) if isfile(join(path, f))]
    # not_exist = []
    # for i in range(1, 20586):
    #     try:
    #         text = open("/Users/kangchieh/Downloads/Bachelorarbeit/corpus_de/test_%s.txt" % i, "r", encoding="utf-8").read()
    #     except FileNotFoundError:
    #         not_exist.append(i)
    #         print(i)
    # # a = [n for n in not_exist if n > 10000]
    # # b = [n for n in a if n < 7000]
    # # print(len(a))
    # # print(len(b))
    # print(len(not_exist))

    df = pd.read_csv("/Users/kangchieh/Downloads/Bachelorarbeit/wiki_concept/filter_de/filter_sim=0.3_freq=0.01_libre_shorten.csv", index_col=0)

    g = GetFeature(df)
    # df1 = g.processing()
    # #
    # df1.to_csv("/Users/kangchieh/Downloads/Bachelorarbeit/wiki_concept/training_de/libre_shorten_v1.csv")
    df = pd.read_csv("/Users/kangchieh/Downloads/Bachelorarbeit/wiki_concept/training_de/libre_shorten_v1.csv", index_col=0)
    # df = pd.read_csv("/Users/kangchieh/Downloads/Bachelorarbeit/wiki_concept/training_de/v3.csv",
    #                  index_col=0)
    #
    t = Training(df)
    t.process(t.df)
    t.f1_score_compare(path="/Users/kangchieh/Downloads/Bachelorarbeit/wiki_concept/training_de/v4.csv")

    #
