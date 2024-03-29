# For calculating mean of a list
import statistics

from matplotlib import pyplot as plt

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


class Training:
    def __init__(self, df, lang, label_de_path, label_en_path):
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
        label_de_path:
            the german topic pairs and the label of if EC is a good expansion of DC (1=good, 0=bad)
        label_en_path:
            the english topic pairs and the label of if EC is a good expansion of DC (1=good, 0=bad)
        X_train:
            training input
        X_test:
            test input
        y_train:
            training output
        y_test:
            test output
        """
        self.df = df
        self.lang = lang
        self.input_X, self.input_y = None, None
        self.label_de = pd.read_csv(label_de_path, index_col=0)
        self.label_en = pd.read_csv(label_en_path, index_col=0)
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
        if lang == 'de':
            df = df.merge(self.label_de, on=['DC', 'EC'])
        elif lang == 'en':
            df = df.merge(self.label_en, on=['DC', 'EC'])
        X, y = self.cleaning_data(df, drop)
        # X = self.building_pipeline(X)
        X = self.gettint_input(X)
        self.input_X, self.input_y = np.array([np.array(x) for x in X["input"].values]), np.array(y.values)
        times = [150]
        for time in times:
            f1_score, tests, preds = [], [], []
            for i in range(time):
                self.X_train, self.X_test, self.y_train, self.y_test = self.split_data(self.input_X, self.input_y)
                # y_test, test_predictions = self.logistic_regression()
                score, test, pred = self.grid_search(model)
                f1_score.append(score)
                tests += test.tolist()
                preds += pred.tolist()
            print("The list of all f1-scores in this training:\n", f1_score)
            precision, recall, _ = precision_recall_curve(tests, preds)
            # pre_recall = PrecisionRecallDisplay.from_predictions(tests, preds)
            plt.plot(precision, recall, label=model)
            plt.legend(loc="upper right")
            plt.xlabel("Precision")
            plt.ylabel("Recall")
            #plt.savefig('/content/drive/MyDrive/test/testing.png')
            print("Mean of f1-score: ", statistics.mean(f1_score))



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
    pass