# string to list
from ast import literal_eval
# wiki, wordnet and word2vec
from HelpFunctions_de import Wiki, Wordnet, Word2vec
# sentiment analysis
from textblob.sentiments import NaiveBayesAnalyzer
from textblob import Blobber
# sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_squared_error
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

    def __init__(self, df):
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
        self.df = df

    def get_wiki(self, df):
        w = Wiki(df)
        return w.processing()

    def get_wordnet(self, df):
        w = Wordnet(df)
        return w.processing()

    def sentiment(self, text):
        tb = Blobber(analyzer=NaiveBayesAnalyzer())
        sen = tb(text).sentiment
        return [sen.p_pos, sen.p_neg]

    def sentiment_analysis(self, df):
        sentiment_dict = {}
        for i in range(len(df)):
            DC, EC = df.at[i, 'DC'], df.at[i, 'EC']
            if DC not in sentiment_dict:
                sentiment_dict[DC] = self.sentiment(DC)
            if EC not in sentiment_dict:
                sentiment_dict[EC] = self.sentiment(EC)

        file = open("/Users/kangchieh/Downloads/Bachelorarbeit/wiki_concept/sentiment.pkl", "wb")
        pickle.dump(sentiment_dict, file)
        file.close()
        # df['DC_Polarity'] = df['DC'].apply(self.sentiment)
        # df['EC_Polarity'] = df['EC'].apply(self.sentiment)
        return df

    def load_sentiment_analysis(self, path, df):
        with open(path, "rb") as f:
            sentiment_dict = pickle.load(f)
        for i in range(len(df)):
            DC, EC = df.at[i, 'DC'], df.at[i, 'EC']
            df['DC_Polarity_diff'] = sentiment_dict[DC][0] - sentiment_dict[DC][1]
            df['EC_Polarity_diff'] = sentiment_dict[EC][0] - sentiment_dict[EC][1]

            return df

    def load_distributional_similarity(self, path, df):
        dsim = pd.read_csv(path, index_col=0)
        df['distributional_similarity'] = dsim['distributional_similarity']

        return df

    def word_embedding(self, df):
        w = Word2vec(df, '/Users/kangchieh/Downloads/Bachelorarbeit/cc.en.100.bin')
        word2vec = w.embedding()
        dc_embedding = []
        ec_embedding = []
        for i in range(len(df)):
            DC, EC = df.at[i, 'DC'], df.at[i, 'EC']
            dc_embedding.append(list(word2vec[DC]))
            ec_embedding.append(list(word2vec[EC]))

        df['DC_embedding'] = dc_embedding
        df['EC_embedding'] = ec_embedding
        return df

    def processing(self):
        wordembed = self.word_embedding(self.df)
        wiki = self.get_wiki(wordembed)
        wordnet = self.get_wordnet(wiki)
        sentiment = self.load_sentiment_analysis(
            "/Users/kangchieh/Downloads/Bachelorarbeit/wiki_concept/training/sentiment.pkl", wordnet)
        final_result = self.load_distributional_similarity(
            "/Users/kangchieh/Downloads/Bachelorarbeit/wiki_concept/concept_wiki_filter_number.csv", sentiment)

        final_result['freq_ratio'] = final_result.apply(
            lambda row: min(row.DC_freq / row.EC_freq, row.EC_freq / row.DC_freq), axis=1)

        return final_result


class Training:
    def __init__(self, df):
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
        self.log_reg = LogisticRegression()
        self.new_df = None
        self.X, self.y = None, None

    def cleaning_data(self):
        """ This function selects only the features needed in the training"""
        self.new_df = self.df.drop(self.df.columns[[2, 3, 4, 5, 7, 8]], 1)
        self.y = self.new_df.pop('good expansion')  # target output
        self.X = self.new_df.iloc[:, 2:]  # training input

        # Since our lists are saved as string we have to transform them back to list
        self.X['DC_embedding'] = self.X['DC_embedding'].apply(literal_eval)
        self.X['EC_embedding'] = self.X['EC_embedding'].apply(literal_eval)

    def building_pipeline(self):
        """ This function scales two columns 'shared_links' and 'shared_categories' with StandardScaler()"""
        pipeline = Pipeline([
            ('std_scaler', StandardScaler()),
        ])
        full_pipeline = ColumnTransformer([
            ("num", pipeline, ['shared_links', 'shared_categories']),
        ])

        self.X[['shared_links', 'shared_categories']] = pd.DataFrame(full_pipeline.fit_transform(self.X))

    def gettint_input(self):
        """ This function gets all training data from the same row to a list"""
        # self.X['shared_links'] = StandardScaler().fit_transform(self.X['shared_links'].values).tolist( )
        # self.X['shared_categories'] = StandardScaler().fit_transform(self.X['shared_categories'].values)
        self.X["input"] = self.X.apply(lambda row: self.row_to_list(row), axis=1)
        self.X["input"] = self.X["input"].apply(lambda x: self.flatten(x))

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

    def logistic_regression(self):
        """ This function performs logistic regression and prints the result at the end"""
        input_X = np.array([np.array(x) for x in self.X["input"].values])
        input_y = np.array(self.y.values)
        X_train, X_test, y_train, y_test = self.split_data(input_X, input_y, 0.2)
        self.log_reg.fit(X_train, y_train)

        test_predictions= self.log_reg.predict(X_test)

        print("Predictions: ", test_predictions)
        print("Labels:      ", y_test)
        print("Sqrt MSE:    ", self.mean_square_error(test_predictions, y_test))
        #rint("f1 Score:    ", self.f1_score(test_predictions, y_test))

    def cross_validation(self, model):
        """
        This function performs cross validation and prints the result at the end
        Parameters
        ----------
        model: sklearn model
            a model to use to fit the data
        """
        input_X = np.array([np.array(x) for x in self.X["input"].values])
        input_y = np.array(self.y.values)
        scores = cross_val_score(model, input_X, input_y,
                                 scoring="neg_mean_squared_error", cv=5)
        tree_rmse_scores = np.sqrt(-scores)
        print("Scores:", tree_rmse_scores)
        print("Mean:  ", tree_rmse_scores.mean())
        print("Standard deviation:", tree_rmse_scores.std())

    def mean_square_error(self, y_test, test_predictions):
        reg_mse = mean_squared_error(y_test, test_predictions)
        reg_rmse = np.sqrt(reg_mse)

        return reg_rmse

    def f1_score(self, y_test, test_predictions):
        f1_score(y_test, test_predictions, average='weighted')


if __name__ == "__main__":
    # data = pardata.load_dataset('wikipedia_oriented_relatedness')
    # print(data)
    # df = pd.read_csv("/Users/kangchieh/Downloads/Bachelorarbeit/wiki_concept/filter/sim=0.3_freq=0.01.csv", index_col=0)
    df = pd.read_csv("/Users/kangchieh/Downloads/Bachelorarbeit/wiki_concept/training/v1.csv", index_col=0)

    # g = GetFeature(df)
    # df1 = g.processing()
    # df1.to_csv("/Users/kangchieh/Downloads/Bachelorarbeit/wiki_concept/training/v1.csv")

    t = Training(df)
    t.cleaning_data()
    t.building_pipeline()
    t.gettint_input()
    t.logistic_regression()
    t.cross_validation(t.log_reg)
