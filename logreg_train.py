#/usr/bin/python3

try:
    import argparse
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.model_selection import train_test_split
except:
    print('[Import error] Please run <pip install -r requirements.txt>')
    exit()

class LogisticRegression(object):
    """
    Linear Regression using Gradient Descent.
    Resulting thetas for model to be used in predict.py
    """
    def __init__(self):
        """
        Class initializer
        """
        self.mean = 0
        self.stdev = 0
        self.theta = np.zeros(2)
    
    def feature_scale_normalise(self, X):
        """
        Normalises & Standardise feature vector X so that
        mean    Xnorm = 0
        stdev   Xnorm = 1
        """
        self.mean = X.mean()
        self.stdev = X.std()
        Xnorm = (X - self.mean) / self.stdev
        print("mean : ", self.mean, "std : ", self.stdev)
        return Xnorm

    def cost(self, X, y, theta):
        """
        Calculates cost for given X and y
        the higher the cost, the more inaccurate the theta values are
        """
        m = X.shape[0]
        prediction = self.hypothesis(X, theta)
        cost = (1/(2*m) * np.sum(np.square(prediction - y)))
        return cost

    def fit(self, X, y, alpha, iter):
        """
        Gradient descent algorithm to update theta values
        """
        X = self.feature_scale_normalise(X)
        m = X.shape[0]
        cost = []
        for i in range(iter):
            loss = self.hypothesis(X, self.theta) - y
            self.theta[0] -= (alpha / m) * np.sum(loss)
            self.theta[1] -= (alpha / m) * np.sum(loss * X)
            cost.append(self.cost(X, y, self.theta))
        print("theta : ", self.theta)
        return self.theta

    def hypothesis(self, X, theta):
        """
        This is valid because X is a single feature vector.
        Otherwise we need to do dot product of X by theta, as well as
        adding a column of 1s in X matrix (to multiply by theta[0])
        """
        ret = X * theta[1] + theta[0]
        # print("hypothesis ret : ", ret)
        return ret

    # def show_data(self, X, y):
    #     """
    #     Plot data (need to adjust once theta calculation is good)
    #     """
    #     Xnorm = self.feature_scale_normalise(X)
    #     plt.plot(X, y, 'b.')
    #     plt.plot(X, self.theta[0] + Xnorm * self.theta[1], 'r-')
    #     # plt.plot()
    #     plt.xlabel("$km$", fontsize=18)
    #     plt.ylabel("$price$", rotation=0, fontsize=18)
    #     plt.legend(['real prices', 'hypothetical prices'], loc='upper right')
    #     plt.show()

    def predict(self, Xval):
        """
        Predict dollar value of car depending on the km given (Xval)
        """
        Xnorm = (Xval - self.mean) / self.stdev
        prediction = self.hypothesis(Xnorm, self.theta)
        return prediction

def parse_arg():
    parser = argparse.ArgumentParser(prog='describe', usage='%(prog)s [-h] datafile.csv', description='Program describing the dataset given.')
    parser.add_argument('datafile', help='the .csv file containing the dataset')
    args = parser.parse_args()
    return args

def read_csv(datafile):
    try:
        f = pd.read_csv(datafile)
        print(f.head())
        print(f.shape)
        print(list(f.columns))
        houses = f['Hogwarts House'].unique()
        y = f['Hogwarts House']
        X = f.drop(['Index','Hogwarts House', 'First Name', 'Last Name', 'Birthday', 'Best Hand'],axis=1)
        X_train, X_test, y_train, y_test = train_test_split(X,y, train_size=0.8,test_size=0.2,random_state=100)
        print(list(houses))
        print(X.shape)
        print(y.shape)
        return X_train, X_test, y_train, y_test, houses
    except:
        raise NameError('[Read error] Wrong file format. Make sure you give an existing .csv file as argument.')

def main():
    try:
        args = parse_arg()
        X_train, X_test, y_train, y_test, houses = read_csv(args.datafile)
        model = LogisticRegression()
        X_norm = model.feature_scale_normalise(X_train)
        # print(X_norm)

    except NameError as e:
        print(e)

if __name__ == '__main__':
    main()