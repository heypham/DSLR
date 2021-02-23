#/usr/bin/python3

try:
    from classes.logisticregression import LogisticRegression
    import pandas as pd
    import matplotlib.pyplot as plt
    from sklearn.model_selection import train_test_split
except NameError as e:
    print(e)
    print('[Import error] Please run <pip install -r requirements.txt>')
    exit()

#     def cost(self, X, y, theta):
#         """
#         Calculates cost for given X and y
#         the higher the cost, the more inaccurate the theta values are
#         """
#         m = X.shape[0]
#         prediction = self.hypothesis(X, theta)
#         cost = (1/(2*m) * np.sum(np.square(prediction - y)))
#         return cost

#     def fit(self, X, y, alpha, iter):
#         """
#         Gradient descent algorithm to update theta values
#         """
#         X = self.feature_scale_normalise(X)
#         m = X.shape[0]
#         cost = []
#         for i in range(iter):
#             loss = self.hypothesis(X, self.theta) - y
#             self.theta[0] -= (alpha / m) * np.sum(loss)
#             self.theta[1] -= (alpha / m) * np.sum(loss * X)
#             cost.append(self.cost(X, y, self.theta))
#         print("theta : ", self.theta)
#         return self.theta

#     def hypothesis(self, X, theta):
#         """
#         This is valid because X is a single feature vector.
#         Otherwise we need to do dot product of X by theta, as well as
#         adding a column of 1s in X matrix (to multiply by theta[0])
#         """
#         ret = X * theta[1] + theta[0]
#         # print("hypothesis ret : ", ret)
#         return ret

#     # def show_data(self, X, y):
#     #     """
#     #     Plot data (need to adjust once theta calculation is good)
#     #     """
#     #     Xnorm = self.feature_scale_normalise(X)
#     #     plt.plot(X, y, 'b.')
#     #     plt.plot(X, self.theta[0] + Xnorm * self.theta[1], 'r-')
#     #     # plt.plot()
#     #     plt.xlabel("$km$", fontsize=18)
#     #     plt.ylabel("$price$", rotation=0, fontsize=18)
#     #     plt.legend(['real prices', 'hypothetical prices'], loc='upper right')
#     #     plt.show()

#     def predict(self, Xval):
#         """
#         Predict dollar value of car depending on the km given (Xval)
#         """
#         Xnorm = (Xval - self.mean) / self.stdev
#         prediction = self.hypothesis(Xnorm, self.theta)
#         return prediction

def main():
    try:
        model = LogisticRegression()
        args = model.parse_arg()
        X, y, features = model.read_csv(args.datafile)
        X_norm = model.feature_scale_normalise(X)
        X_train, X_test, y_train, y_test = model.split_data(X_norm, y)

    except NameError as e:
        print(e)

if __name__ == '__main__':
    main()