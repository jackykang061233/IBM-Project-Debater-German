# For calculating mean of a list
import statistics

from matplotlib import pyplot as plt
from HelpFunctions_de import Wiki, Wordnet_de, Word2vec, Sentiment_Analysis

# sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_validate, train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_squared_error, precision_score, recall_score, precision_recall_curve, f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestRegressor
# Test set stat
from scipy import stats
# Basic functions
import pickle
import numpy as np
import pandas as pd


class GetFeature:
    """
    A class used to get the feature of the german model, other training features like distributional similarity are
    already included in the dataframe before performing this class

    Methods
    -------
    get_wordnet(df):
        This method gets the four features from Germanet
    get_sentiment(df, path):
        This method gets the sentiment of the given topics
    word_embedding(df):
        This method gets the word embedding of the given topics
    process(df, sentiment_path=None):
        This method performs the final process of get the training features
    """

    def __init__(self):
        pass

    # def get_wiki(self, df, path_cat, path_link):
    #     w = Wiki(df)
    #     w.processing(path_cat, path_link)
    #     with open("/Users/kangchieh/Downloads/Bachelorarbeit/wiki_concept/wiki/cat_de.pkt", "rb") as f:
    #         cat = pickle.load(f)
    #     with open("/Users/kangchieh/Downloads/Bachelorarbeit/wiki_concept/wiki/link_de.pkt", "rb") as f:
    #         link = pickle.load(f)
    #     df['shared_categories'] = df.apply(lambda row: cat.get((row.DC, row.EC)), axis=1)
    #     df['shared_links'] = df.apply(lambda row: link.get((row.DC, row.EC)), axis=1)
    #     return df

    def get_wordnet(self, df):
        """
        This method gets the four features from Germanet

        Parameters
        ----------
        df: Dataframe
            a dataframe with topic pairs and other training features

        Returns
        -------
        Dataframe
            the input dataframe plus the training features from Germanet
        """
        w = Wordnet_de(df, "/Users/kangchieh/Downloads/Bachelorarbeit/germanet/GN_V160/GN_V160_XML")
        df_wordnet = w.processing()
        return df_wordnet

    def get_sentiment(self, df, path):
        """
        This method gets the sentiment of the given topics

        Parameters
        ----------
        df: Dataframe
            a dataframe with topic pairs and other training features
        path: String
            the path of a dictionary with already saved topics and their sentiment

        Returns
        -------
        Dataframe
            the input dataframe plus the sentiment features
        """
        s = Sentiment_Analysis(df)
        s.processing(path)
        with open(path, "rb") as f:
            sentiment = pickle.load(f)
        df['DC_sentiment'] = df["DC"].apply(lambda x: sentiment.get(x))
        df['EC_sentiment'] = df["EC"].apply(lambda x: sentiment.get(x))
        return df

    def word_embedding(self, df):
        """
        This method gets the word embedding of the given topics

        Parameters
        ----------
        df: Dataframe
            a dataframe with topic pairs and other training features

        Returns
        -------
        Dataframe
            the input dataframe plus the word embedding of every topic
        """
        w = Word2vec(df, '/Users/kangchieh/Downloads/Bachelorarbeit/cc.de.100.bin', 'de')
        word2vec = w.embedding_fasttext()  # use fasttext embedding
        dc_embedding = []
        ec_embedding = []
        for i in range(len(df)):
            DC, EC = df.at[i, 'DC'], df.at[i, 'EC']
            dc_embedding.append(list(word2vec[DC]))
            ec_embedding.append(list(word2vec[EC]))

        df['DC_embedding'] = dc_embedding
        df['EC_embedding'] = ec_embedding
        return df

    def processing(self, df, sentiment_path=None):
        """
        This method performs the final processing of get the training features

        Parameters
        ----------
        df: Dataframe
            a dataframe with topic pairs and other training features
        sentiment_path: String
            the path of a dictionary with already saved topics and their sentiment

        Returns
        -------
        Dataframe
            the dataframe plus the word embedding of every topic
        """
        wordnet = self.get_wordnet(df)
        final_result = self.get_sentiment(wordnet, sentiment_path)

        final_result['freq_ratio'] = final_result.apply(
            lambda row: min(row.DC_freq / row.EC_freq,
                            row.EC_freq / row.DC_freq) if row.DC_freq != 0 and row.EC_freq != 0 else 1 / max(
                row.DC_freq, row.EC_freq), axis=1)  # calculating frequency ratio of two words

        return final_result


class Training:
    def __init__(self, df, lang):
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
            12. hypernym
            13. hyponym
            14. co-hypernym
            15. synonym
            16. DC_Polarity_diff
            17. EC_Polarity_diff
            18. freq_ratio
        lang: String
            The language of the model, e.g. 'de' for german and 'en' for english
        input_X: Dataframe
            A input X is predefined as None but will later be assigned to our training data
        Input_Y: Series
            Y is predefined as None but will later be assigned to our output target
        label_de:
        
        label_en:
        X_train:
        X_test:
        y_train:
        y_test:
        """
        self.df = df
        self.lang = lang
        self.input_X, self.input_y = None, None
        self.label_de = pd.read_csv("/Users/kangchieh/Downloads/Bachelorarbeit/wiki_concept/expansion_label_de.csv",
                                 index_col=0)
        self.label_en = pd.read_csv("/Users/kangchieh/Downloads/Bachelorarbeit/wiki_concept/expansion_label_en.csv",
                                    index_col=0)
        self.X_train, self.X_test, self.y_train, self.y_test = None, None, None, None

    def cleaning_data(self, df, drop=None):
        """ This function selects only the features needed in the training"""
        if drop is None:
            drop = []
        X = df.drop(['DC', 'EC', 'stop_words', 'substring', 'DC_freq', 'EC_freq', 'filter out', 'DC_embedding',
                     'EC_embedding'] + drop, axis=1)
        y = X.pop('good expansion')  # target output

        return X, y

    # def building_pipeline(self, df):
    #     """ This function scales two columns 'shared_links' and 'shared_categories' with StandardScaler()"""
    #     pipeline = Pipeline([
    #         ('std_scaler', StandardScaler()),
    #     ])
    #     full_pipeline = ColumnTransformer([
    #         ("num", pipeline, ['shared_links', 'shared_categories']),
    #     ])
    #     df[['shared_links', 'shared_categories']] = pd.DataFrame(full_pipeline.fit_transform(df))
    #     return df

    def gettint_input(self, df):
        """ This function gets all training data from the same row to a list"""
        # self.X['shared_links'] = StandardScaler().fit_transform(self.X['shared_links'].values).tolist( )
        # self.X['shared_categories'] = StandardScaler().fit_transform(self.X['shared_categories'].values)
        df["input"] = df.apply(lambda row: self.row_to_list(row), axis=1)
        df["input"] = df["input"].apply(lambda x: self.flatten(x))
        return df

    def row_to_list(self, row):
        """
        This method make a Dataframe row values to one list

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

    def split_data(self, X, y, split_ratio=0.1):
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
        return train_test_split(X, y, test_size=split_ratio, stratify=y)

    def logistic_regression(self):
        """ This function performs logistic regression and prints the result at the end"""
        log_reg = LogisticRegression(max_iter=5000)
        log_reg.fit(self.X_train, self.y_train)

        test_predictions = log_reg.predict(self.X_test)
        eval_interval = self.evaluation(test_predictions, self.y_test)

        print('Logistic Regression:')
        print("Predictions:   ", test_predictions)
        print("Labels:        ", self.y_test)
        print("\nSqrt MSE:      ", self.mean_square_error(test_predictions, self.y_test))
        print("f1 Score:      ", self.f1_score(self.y_test, test_predictions))
        print("Precision:     ", precision_score(self.y_test, test_predictions, average="weighted"))
        print("Recall:        ", recall_score(self.y_test, test_predictions, average="weighted"))
        print("Errors Interval", eval_interval)
        return self.y_test, test_predictions

    def cross_validation(self, model):
        """
        This function performs cross validation and prints the result at the end
        Parameters
        ----------
        model: sklearn model
            a model to use to fit the data
        """
        scores = cross_validate(model, self.X_train, self.y_train,
                                scoring=("f1", "precision", "recall"), cv=5)
        print("\nCross Validation:")
        print("Scores:", scores)
        # print("Mean:  ", scores.mean())
        # print("Standard deviation:", scores.std())
        return scores['test_f1'].mean()

    def mean_square_error(self, y_test, test_predictions):
        reg_mse = mean_squared_error(y_test, test_predictions)
        reg_rmse = np.sqrt(reg_mse)

        return reg_rmse

    def f1_score(self, y_test, test_predictions):
        return f1_score(y_test, test_predictions, average='macro')

    def decision_tree(self):
        # Decision tree using cross validation
        decision_tree_reg = DecisionTreeClassifier()
        self.cross_validation(decision_tree_reg)

    def random_forest_regressor(self):
        random_forest_reg = RandomForestRegressor()
        self.cross_validation(random_forest_reg)

    def evaluation(self, predictions, y_test):
        confidence = 0.95
        squared_errors = (predictions - y_test) ** 2
        interval = np.sqrt(stats.t.interval(confidence, len(squared_errors) - 1,
                                            loc=squared_errors.mean(), scale=stats.sem(squared_errors)))
        print(interval)

    def grid_search(self, model):
        if model == 'logistic':
            grid_parameters = {
                'C': [10, 100, 1000],
                # 'max_iter': list(range(1000, 2200, 250))
            }
            log_reg = LogisticRegression(C=100, max_iter=5000)

            grid_search = GridSearchCV(log_reg, grid_parameters, cv=5,
                                       scoring='f1_macro',
                                       return_train_score=True)

        elif model == 'decision':
            grid_parameters = [{
                'criterion': ['gini', 'entropy'],
                'max_depth': [2, 4, 6, 8, 10],
                'max_features': ["auto", "sqrt", "log2"],
                'splitter': ["best", "random"]
            }]
            decision_tree = DecisionTreeClassifier()

            grid_search = GridSearchCV(decision_tree, grid_parameters, cv=5,
                                       scoring='f1_macro',
                                       return_train_score=True)
        else:
            raise ValueError("The model can't be found!")

        # print("Best Parameters: \n{}\n".format(grid_search.best_estimator_))
        grid_search.fit(self.X_train, self.y_train)
        best_model = grid_search.best_estimator_
        # if model == 'logistic':
        #   for param in self.parameters_logistic:
        #       self.parameters_logistic[param][best_model.get_params()[param]] += 1
        # elif model == 'decision':
        #   for param in self.parameters_decision:
        #       self.parameters_decision[param][best_model.get_params()[param]] += 1
        pred = best_model.predict(self.X_test)

        score = self.f1_score(self.y_test, pred)

        return score, self.y_test, pred

    # def precision_recall(self):

    def f1_score_compare(self, lang='de', drop=None,  path=None, df=None):
        f1_score_1 = []
        if path is not None:
            df = pd.read_csv(path, index_col=0)
        f1_score_2 = []
        score_1 = self.process(self.df, drop, self.lang, 'decision')
        score_2 = self.process(df, drop, lang, 'decision')
        f1_score_1.append(score_1)
        f1_score_2.append(score_2)
        # for i in range(times):
        #     score_1 = self.process(self.df, drop, self.lang, 'logistic')
        #     score_2 = self.process(df, drop, lang, 'decision')
        #     # y_test_1, test_predictions_1 = self.process(self.df, drop, self.lang)
        #     # y_test_2, test_predictions_2 = self.process(df, drop, lang)
        #     f1_score_1.append(score_1)
        #     f1_score_2.append(score_2)
        print(f1_score_1)
        print(f1_score_2)
        # plt.plot(f1_score_1, label='1')
        # plt.plot(f1_score_2, label='2')
        # plt.xlabel("Times")
        # plt.ylabel("f1 score")
        # plt.legend(loc="upper left", fontsize=10)
        # plt.show()

    def process(self, df, drop, lang, model):
        if lang == 'de':
            df = df.merge(self.label_de, on=['DC', 'EC'])
        elif lang == 'en':
            df = df.merge(self.label_en, on=['DC', 'EC'])

        df.to_csv("/Users/kangchieh/Downloads/Bachelorarbeit/wiki_concept/test.csv")

        X, y = self.cleaning_data(df, drop)
        # X = self.building_pipeline(X)
        X = self.gettint_input(X)
        self.input_X, self.input_y = np.array([np.array(x) for x in X["input"].values]), np.array(y.values)
        self.X_train, self.X_test, self.y_train, self.y_test = self.split_data(self.input_X, self.input_y)
        # y_test, test_predictions = self.logistic_regression()
        if model == 'logistic':
            score = self.cross_validation(LogisticRegression())
        elif model == 'decision':
            score = self.cross_validation(DecisionTreeClassifier())

        return score

    def process_grid_search(self, df, lang, model, drop=None):
        # if lang == 'de':
        #     df = df.merge(self.label, on=['DC', 'EC'])
        # elif lang == 'en':
        #     df = df.merge(self.label_en, on=['DC', 'EC'])

        # df.to_csv("/Users/kangchieh/Downloads/Bachelorarbeit/wiki_concept/test.csv")

        X, y = self.cleaning_data(df, drop)
        # X = self.building_pipeline(X)
        X = self.gettint_input(X)
        self.input_X, self.input_y = np.array([np.array(x) for x in X["input"].values]), np.array(y.values)
        # times = [10, 50, 100, 250, 500, 1000]
        times = [150]
        for time in times:
            f1_score, tests, preds = [], [], []
            for i in range(time):
                if i == 75 or i == 200:
                    print(i)
                self.X_train, self.X_test, self.y_train, self.y_test = self.split_data(self.input_X, self.input_y)
                # y_test, test_predictions = self.logistic_regression()
                score, test, pred = self.grid_search(model)
                f1_score.append(score)
                tests += test.tolist()
                preds += pred.tolist()
            print(f1_score)
            precision, recall, _ = precision_recall_curve(tests, preds)
            # pre_recall = PrecisionRecallDisplay.from_predictions(tests, preds)
            plt.plot(precision, recall, label=model)
            plt.legend(loc="upper right")
            plt.xlabel("Precision")
            plt.ylabel("Recall")
            plt.savefig('/content/drive/MyDrive/test/testing.png')
            print(statistics.mean(f1_score))

        return score

    def final_training_process(self):
        df = pd.read_csv('/content/drive/MyDrive/test/test_fasttext.csv')
        training = Training(df, 'de')
        _, pre_recall_1 = training.process_grid_search(training.df, 'de', 'logistic')
        df = pd.read_csv('/content/drive/MyDrive/test/test_statified.csv')
        training = Training(df, 'de')
        _, pre_recall_2 = training.process_grid_search(training.df, 'de', 'logistic')

    # X, y = self.cleaning_data(df, drop)
        # X = self.building_pipeline(X)
        # X = self.gettint_input(X)
        # self.input_X, self.input_y = np.array([np.array(x) for x in X["input"].values]), np.array(y.values)
        # times = [10, 50, 100, 250, 500, 1000]
        #
        # for time in times:
        #     f1_score = []
        #     for i in range(time):
        #         self.X_train, self.X_test, self.y_train, self.y_test = self.split_data(self.input_X, self.input_y)
        #         # y_test, test_predictions = self.logistic_regression()
        #         score = self.grid_search(model)
        #         f1_score.append(score)
        #     print(statistics.mean(f1_score))
        #
        # return score


if __name__ == "__main__":
    df = pd.read_csv("/Users/kangchieh/Downloads/Bachelorarbeit/wiki_concept/test/test_%s.csv" % 'fasttext', index_col=0)
    print(len(df))
    print(len(df[df['good expansion']==1]))

    #
