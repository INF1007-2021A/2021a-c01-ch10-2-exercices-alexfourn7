#!/usr/bin/env python
# -*- coding: utf-8 -*-

# TODO: Importez vos modules ici
import os
import csv
import numpy as np
import pandas as pd
from sklearn import model_selection
from sklearn import ensemble
from sklearn import linear_model
import matplotlib.pyplot as plt
from sklearn import metrics

# TODO: Définissez vos fonctions ici


def read_csv(chemin, fichier):
    with open(os.path.join(chemin, fichier), 'r') as f:
        data = pd.read_csv(f, delimiter=";")
    return data


def data(dataframe):
    y = dataframe['quality']
    X = dataframe.drop('quality', axis=1)
    return X, y


def sets(X, y):
    X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.2)
    return X_train, X_test, y_train, y_test


def model_random_forest(X_train, y_train):
    model = ensemble.RandomForestRegressor()
    model.fit(X_train, y_train)
    return model


def model_linear_regression(X_train, y_train):
    model = linear_model.LinearRegression()
    model.fit(X_train, y_train)
    return model


def tests(model1, model2, X_test, y_test):
    score_1 = model1.score(X_test, y_test)
    score_2 = model2.score(X_test, y_test)

    predict_1 = model1.predict(X_test)
    predict_2 = model2.predict(X_test)

    return score_1, predict_1, score_2, predict_2


def graphique_random(y_test, predict_random):
    plt.plot(y_test.to_numpy(), label='Target values')
    plt.plot(predict_random, label='Predicted values')
    plt.legend(loc="upper left")
    plt.title("RandomForestRegressor predictions analysis")
    plt.xlabel("Number of samples")
    plt.ylabel("Quality")
    plt.show()


def graphique_linear(y_test, predict_linear):
    plt.plot(y_test.to_numpy(), label='Target values')
    plt.plot(predict_linear, label='Predicted values')
    plt.legend(loc="upper left")
    plt.title("LinearRegression predictions analysis")
    plt.xlabel("Number of samples")
    plt.ylabel("Quality")
    plt.show()

def mse(y_true, y_pred):
    return metrics.mean_squared_error(y_true, y_pred)

if __name__ == '__main__':
    # TODO: Appelez vos fonctions ici
    d = read_csv("data", "winequality-white.csv")
    X, y = data(d)
    X_train, X_test, y_train, y_test = sets(X, y)

    random_model = model_random_forest(X_train, y_train)
    linear_model = model_linear_regression(X_train, y_train)

    score_random, predict_random, score_linear, predict_linear = tests(random_model, linear_model, X_test, y_test)

    print(f" Score Random Forest: {score_random}\n Score Linear Regression: {score_linear}")

    graphique_random(y_test, predict_random)
    graphique_linear(y_test, predict_linear)

    mse_random = mse(y_test, predict_random)
    mse_linear = mse(y_test, predict_linear)
    print(f" MSE Random Forest: {mse_random}\n MSE Linear Regression: {mse_linear}")
