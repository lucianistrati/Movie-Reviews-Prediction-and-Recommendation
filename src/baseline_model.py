from sklearn.feature_extraction.text import CountVectorizer
from spacy.lang.fr.stop_words import STOP_WORDS as fr_stop_words
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import BernoulliNB, MultinomialNB
from sklearn.metrics import accuracy_score, f1_score
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import AdaBoostClassifier, AdaBoostRegressor
from sklearn.linear_model import Perceptron, Ridge, Lasso, SGDClassifier, \
    SGDRegressor, LinearRegression, LogisticRegression
from sklearn.neural_network import MLPRegressor
from xgboost import XGBClassifier, XGBRegressor
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.svm import SVC, SVR

import pandas as pd
import numpy as np

import os

def train_baselines_model(X_train, y_train, X_test, y_test):
    best_mae = 1e18
    best_model = ""
    models = [SVC(class_weight="balanced"), SVR(), BernoulliNB(),
              MultinomialNB(), Perceptron(class_weight="balanced"),
              AdaBoostClassifier(), RandomForestClassifier(class_weight="balanced"),
              DecisionTreeClassifier(class_weight="balanced"), XGBClassifier(class_weight="balanced"),
              DecisionTreeRegressor(), RandomForestRegressor(), XGBRegressor(),
              Ridge(), Lasso(), MLPRegressor(),
              AdaBoostRegressor(), KNeighborsClassifier(),
              KNeighborsRegressor(),
              SGDClassifier(class_weight="balanced"), SGDRegressor(), LinearRegression(),
              LogisticRegression(class_weight="balanced")]

    models_names = ["SVC()", "SVR()", "BernoulliNB()", "MultinomialNB()",
                    "Perceptron()", "AdaBoostClassifier()",
                    "RandomForestClassifier()",
                    "DecisionTreeClassifier()", "XGBClassifier()",
                    "DecisionTreeRegressor()", "RandomForestRegressor()",
                    "XGBRegressor()", "Ridge()", "Lasso()",
                    "MLPRegressor()", "AdaBoostRegressor()",
                    "KNeighborsClassifier()", "KNeighborsRegressor()",
                    "SGDClassifier()", "SGDRegressor()",
                    "LinearRegression()", "LogisticRegression()"]

    for idx, (model, model_name) in enumerate(zip(models, models_names)):
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)

        mae = mean_absolute_error(y_pred, y_test)

        print("***"*10)
        print(model_name)
        print("MAE:", mae)
        print("BEST MAE:", best_mae, "best mdel: ", best_model, idx)
        if mae < best_mae:
            best_mae = mae
            best_model = model_name

    print("###")
    print(best_model, best_mae)


from score_predictions import score

def finetune_best_model(X_train, y_train, X_test, y_test=None):
    model = LogisticRegression(class_weight="balanced")

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    print(accuracy_score(y_pred, y_test))
    if y_test is not None:
        mae = mean_absolute_error(y_pred, y_test)
        print(mae)




def main():
    data_filenames = ["train.csv", "test.csv"]
    test_submission = False
    fit_both_sets = False
    for filename in data_filenames:
        df = pd.read_csv(os.path.join("data", filename))
        if test_submission:
            if filename == "test.csv":
                df = pd.read_csv(os.path.join("test", "test.csv"))
            if filename == "train.csv":
                df = pd.read_csv(os.path.join("data", "data.csv"))
        if filename.startswith("train"):
            X_train = np.array(df['text'].tolist())
            y_train = np.array(df['label'].tolist())
        else:
            X_test = np.array(df['text'].tolist())
            if test_submission:
                y_test = None
            else:
                y_test = np.array(df['label'].tolist())

    cv = CountVectorizer()

    if fit_both_sets:
        X = np.concatenate((X_train, X_test))
        cv.fit(X)
        X_train = cv.transform(X_train)
        X_test = cv.transform(X_test)
    else:
        X_train = cv.fit_transform(X_train)
        X_test = cv.transform(X_test)

    # print(train_baselines_model(X_train, y_train, X_test, y_test))
    finetune_best_model(X_train, y_train, X_test, y_test)

if __name__=="__main__":
    main()




