import argparse
import numpy as np
import pandas as pd
from classes.feature import Feature

class LogisticRegression(object):
    """
    Logistic Regression class using Gradient Descent
    """
    def __init__(self):
        """
        Class initializer
        """
        self.mean = 0
        self.stdev = 0
        self.theta = np.zeros(2) # Need to change number of theta values to number of features + 1
        self.feature_calc = None

    def parse_arg(self):
        parser = argparse.ArgumentParser(prog='describe', usage='%(prog)s [-h] datafile.csv', description='Program describing the dataset given.')
        parser.add_argument('datafile', help='the .csv file containing the dataset')
        args = parser.parse_args()
        return args
    
    def read_csv(self, datafile):
        """
        Function to read csv file and split into training/testing sets
        returns train/test sets for X and y + list of 4 houses
        """
        try:
            f = pd.read_csv(datafile)
            houses = f['Hogwarts House'].unique()
            features = list(f.columns[6:])
            y = f['Hogwarts House']
            X = f.drop(['Index','Hogwarts House', 'First Name', 'Last Name', 'Birthday', 'Best Hand'],axis=1)
            
            # Transform arrays as numpy arrays for calculations
            X = np.array(X).T
            y = np.array([y])
            features = np.array(features).T
            return X, y, features
        except:
            raise NameError('[Read error] Wrong file format. Make sure you give an existing .csv file as argument.')

    def split_data(self, X, y):
        X_train, X_test, y_train, y_test = train_test_split(X,y, train_size=0.8,test_size=0.2,random_state=100)
        return X_train, X_test, y_train, y_test