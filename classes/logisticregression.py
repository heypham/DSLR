import argparse
import numpy as np
import pandas as pd
from classes.feature import Feature
from sklearn.model_selection import train_test_split

class LogisticRegression(object):
    """
    Logistic Regression class using Gradient Descent
    """
    def __init__(self):
        """
        Class initializer
        """
        self.X = []
        self.y = []
        self.features = []
        self.mean = []
        self.stdev = []
        self.theta = np.zeros(14).reshape(-1,1) # Need to change number of theta values to number of features + 1 (for bias)

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
            # print(houses)
            features = list(f.columns[6:])
            y = f['Hogwarts House']
            X = f.drop(['Index','Hogwarts House', 'First Name', 'Last Name', 'Birthday', 'Best Hand'],axis=1)
            
            # Transform arrays as numpy arrays for calculations
            self.X = np.array(X).T
            self.y = np.array([y])
            self.features = np.array(features).T
            return self.X, self.y, self.features
        except:
            raise NameError('[Read error] Wrong file format. Make sure you give an existing .csv file as argument.')

    def split_data(self, X, y):
        """
        Splitting dataset into training data/testing data
        """
        X_train, X_test, y_train, y_test = train_test_split(X.T,y.T, train_size=0.8,test_size=0.2,random_state=100)
        return X_train, X_test, y_train, y_test

    def remove_empty_values(self, X):
        x_filtered = []
        for x in X:
            if x == x:
                x_filtered.append(x)
        return x_filtered

    def describe(self, features_names, X):
        """
        returns a Feature object list with each
        mean, std, quartiles, min & max values
        """
        i = 0
        features = []
        for x in X:
            feature = Feature(features_names[i], self.remove_empty_values(x))
            features.append(feature)
            i +=1
        return features

    def feature_scale_normalise(self, X):
        """
        Normalises & Standardise each feature vector in X
        using Feature class
        """
        Xnorm = []
        features_describe = self.describe(self.features, X)
        for i in range(len(X)):
            self.mean.append(features_describe[i].mean)
            self.stdev.append(features_describe[i].std)
            Xnorm.append((X[i] - self.mean[i]) / self.stdev[i])
        Xnorm = np.array(Xnorm)
        Xnorm = Xnorm.T
        Xnorm = np.insert(Xnorm, 0, 1, axis=1)
        return Xnorm

    def one_hot_encoding(self, y):
        """
        Changes each y into 1x4 matrix
        Ravenclaw =  [ 1 0 0 0 ]
        Slytherin =  [ 0 1 0 0 ]
        Gryffindor = [ 0 0 1 0 ]
        Hufflepuff = [ 0 0 0 1 ]
        """
        y_encoded = []
        count = 0
        for y_i in y:
            if y_i == "Ravenclaw":
                y_encoded.append([1, 0, 0, 0])
            elif y_i == "Slytherin":
                y_encoded.append([0, 1, 0, 0])
            elif y_i == "Gryffindor":
                y_encoded.append([0, 0, 1, 0])
            elif y_i == "Hufflepuff":
                y_encoded.append([0, 0, 0, 1])
        y_encoded = np.array(y_encoded)
        return y_encoded

    def pre_activation(self, X, theta):
        """
        Dot product of weights (θ vector) and features (X)
        will be used in activation (sigmoid)
        """
        ret = np.dot(X, theta)
        return ret

    def activation(self, z):
        """
        The activated function is used in the hypothesis.
        Here we use sigmoid function
        """
        return 1 / (1 + np.exp(-z))

    def hypothesis(self, features, weights):
        """
            Predict the class
            **input: **
                *features: (Numpy Matrix)
                *weights: (Numpy vector)
            **reutrn: (Numpy vector)**
                *0 or 1
        """
        z = self.pre_activation(features, weights)
        y = self.activation(z)
        return np.round(y)

    def cost(self, X, y, theta):
        """
        Calculates cost for given X and y
        the higher the cost, the more inaccurate the theta values are
        """
        m = X.shape[0]
        prediction = self.activation(X, theta)
        cost = (1/(2*m) * np.sum(np.square(prediction - y)))
        return cost

    def fit(self, X, y, alpha, iter):
        """
        Gradient descent algorithm to update theta values
        """
        X = self.feature_scale_normalise(X)
        m = X.shape[0]
        cost = []
        for _ in range(iter):
            loss = self.activation(X, self.theta) - y
            self.theta[0] -= (alpha / m) * np.sum(loss)
            self.theta[1] -= (alpha / m) * np.sum(loss * X)
            cost.append(self.cost(X, y, self.theta))
        print("theta : ", self.theta)
        return self.theta