# -*- coding: utf-8 -*-
"""
Created on Tue Dec 14 19:44:03 2021

@author: dovyd
"""

import pandas as pd
import numpy as np
import sklearn
from sklearn import linear_model
from sklearn.utils import shuffle
import matplotlib.pyplot as pyplot
import pickle
from matplotlib import style
from tqdm import tqdm

data = pd.read_csv("student-mat.csv", sep=";")


data = data[["G1", "G2", "G3", "studytime", "failures", "absences"]]

#print(data.head())
predict = "G3"

X = np.array(data.drop([predict], 1))
y = np.array(data[predict])
x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size = 0.1)

best = 0
def train_and_save_model(best):
    #train model
    for i in tqdm (range (1000), desc="Loading..."):
        x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size = 0.1)
        linear = linear_model.LinearRegression()
        
        linear.fit(x_train, y_train)
        acc = linear.score(x_test, y_test)
        
        #print(i , " ", acc)
        
        #save model
        if acc > best:
            best = acc
            with open("studentmodel.pickle", "wb") as f:
                pickle.dump(linear, f)
    return best
                

#best = train_and_save_model(best)
print(f"\nBest: {best}")

def print_predictions():
    pickle_in = open("studentmodel.pickle", "rb")
    linear = pickle.load(pickle_in)
    
    print("Coefficient: \n", linear.coef_)
    print("Intercept: \n", linear.intercept_)
    
    predictions = linear.predict(x_test)
    good_predictions = 0
    bad_predictions = 0
    for x in range(len(predictions)):
        if round(predictions[x]) == y_test[x]:
            good_predictions = good_predictions + 1
        else:
            bad_predictions = bad_predictions + 1
        print("Predictions: ", predictions[x], x_test[x], "Real: ", y_test[x])
    print(f"All predictions: {x+1}")
    print(f"Good predictions: {good_predictions}")
    print(f"Bad predictions: {bad_predictions}")
        
    p = "G1"
    style.use("ggplot")
    pyplot.scatter(data[p], data["G3"])
    pyplot.xlabel(p)
    pyplot.ylabel("Final grade")
    pyplot.show()
    
print_predictions()
    
    
    
    
    
    
    
    