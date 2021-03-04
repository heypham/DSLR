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
        self.houses = ['Gryffindor', 'Slytherin', 'Hufflepuff', 'Ravenclaw']
        self.colors = {
            'Gryffindor': 'red',
            'Slytherin': 'green',
            'Hufflepuff': 'yellow',
            'Ravenclaw': 'blue'
        }
        self.X = []
        self.y = []
        self.features = []
        self.mean = []
        self.stdev = []
        self.thetas = np.zeros((14, 4))

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

    def clean_data(self, X, y):
        X_clean = []
        y_clean = []
        for i in range(X.shape[0]):
            empty_value = 0
            for j in range(X.shape[1]):
                if X[i][j] != X[i][j]:
                    empty_value = 1
            if empty_value == 0:
                X_clean.append(X[i])
                y_clean.append(y[0][i])
        X_clean = np.array(X_clean)
        y_clean = np.array(y_clean).reshape(-1,1)
        return X_clean.T, y_clean

    def split_data(self, X, y):
        """
        Splitting dataset into training data/testing data
        """
        X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8,test_size=0.2,random_state=100)
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
        Gryffindor =  [ 1 0 0 0 ]
        Slytherin =  [ 0 1 0 0 ]
        Hufflepuff = [ 0 0 1 0 ]
        Ravenclaw = [ 0 0 0 1 ]
        """
        y_encoded = []
        count = 0
        encoding = {
            'Gryffindor': [1, 0, 0, 0],
            'Slytherin': [0, 1, 0, 0],
            'Hufflepuff': [0, 0, 1, 0],
            'Ravenclaw': [0, 0, 0, 1]
        }
        for y_i in y:
            y_encoded.append(encoding[y_i[0]])
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

    def hypothesis(self, X, weights):
        """
            Predict the class
            **input: **
                *X: (Numpy Matrix)
                *weights: (Numpy vector)
            **reutrn: (Numpy vector)**
                *0 or 1
        """
        Z = self.pre_activation(X, weights)
        G = self.activation(Z)
        return G

    def cost(self, X, y, theta):
        """
        Calculates cost for given X and y
        the higher the cost, the more inaccurate the theta values are
        """
        m = X.shape[0]
        prediction = self.activation(X, theta)
        cost = (1/(2*m) * np.sum(np.square(prediction - y)))
        """
        messy dirty cost to rewrite
        """
        H = model.hypothesis(X_norm, model.theta)
        v1 = np.log(np.ones(H.shape) - H)
        y_griffindor = y_encoded[:,0]
        v2 = (np.ones(y_griffindor.shape) - y_griffindor).reshape(-1,1)
        v3 = np.dot(v2.T, v1)
        v4 = np.dot(y_griffindor, np.log(H)).reshape(-1,1)
        cost = (-1/H.shape[0]) * (v4[0] + v3[0])
        return cost

    def fit(self, X, y, alpha, iter):
        """
        Gradient descent algorithm to update theta values
        """
        m = X.shape[0]
        cost = []
        for _ in range(iter):
            loss = (self.hypothesis(X, self.thetas) - y).T
            loss_per_feature = np.dot(loss, X).T
            self.thetas -= alpha * (1/m) * loss_per_feature
        return self.thetas
            # cost.append(self.cost(X, y, self.theta))
        # return self.thetas, cost

    def H_from_probability_to_absolute_values(self, X):
        m = X.shape[0]
        H_absolute = []
        H_pobability = self.hypothesis(X, self.thetas)
        for i in range(m):
            H_absolute.append([0, 0, 0, 0])
            H_absolute[i][np.argmax(H_pobability[i])] = 1
        H_absolute = np.array(H_absolute)
        return H_absolute
