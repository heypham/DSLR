#/usr/bin/python3

try:
    import numpy as np
    import pandas as pd
    from classes.feature import Feature
    from sklearn.model_selection import train_test_split
except NameError as e:
    print(e)
    print('[Import error] Please run <pip install -r requirements.txt>')
    exit()

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
        self.verbose = 0
        self.X = []
        self.y = []
        self.features = []
        self.mean = []
        self.stdev = []
        self.thetas = np.zeros((14, 4))
        self.cost_history = []

    def parse_arg(self):
        try:
            parser = argparse.ArgumentParser(prog='describe', usage='%(prog)s [-h] datafile.csv', description='Program describing the dataset given.')
            parser.add_argument('datafile', help='the .csv file containing the dataset')
            args = parser.parse_args()
            return args
        except:
            raise NameError('[Parse error] There has been an error while parsing the arguments.')

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
            self.X = np.array(X).T
            self.y = np.array([y])
            self.features = np.array(features).T
            if self.verbose > 0:
                print('\n-->\tReading CSV file.')
            return self.X, self.y, self.features
        except:
            raise NameError('[Read error] Wrong file format. Make sure you give an existing .csv file as argument.')

    def set_verbose(self, verbose):
        if verbose:
            self.verbose = verbose

    def clean_data(self, X, y):
        try:
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
            if self.verbose > 0:
                print('\n-->\tCleaning dataset.')
            if self.verbose > 1:
                print('\tThe rows with empty values have been removed.')
            return X_clean.T, y_clean
        except:
            raise NameError('[Process error] There has been an error while cleaning the data.')

    def split_data(self, X, y, train_percentage):
        """
        Splitting dataset into training data/testing data
        """
        try:
            train = float(train_percentage)
            test = float(1 - train)
            X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=train, test_size=test, random_state=100)
            if self.verbose > 0:
                print('\n-->\tSplitting dataset')
            if self.verbose > 1:
                print('\t{:.2f} % train\n\t{:.2f} % test'.format(train * 100, test * 100))
            return X_train, X_test, y_train, y_test
        except:
            raise NameError('[Process error] There has been an error while splitting the dataset.')

    def fill_data(self, X, mean):
        """
        For predictor, filling empty data with corresponding feature mean so it doesn't affect the result.
        """
        try:
            X_filled = []
            num_features = X.shape[0]
            len_data = X.shape[1]
            for i in range(num_features):
                current_feature = []
                for j in range(len_data):
                    if X[i][j] != X[i][j]:
                        x_i_j = mean[i]
                    else:
                        x_i_j = X[i][j]
                    current_feature.append(x_i_j)
                X_filled.append(current_feature)
            X_filled = np.array(X_filled)
            return X_filled
        except:
            raise NameError('[Process error] The dataset cannot be filled.')

    def remove_empty_values(self, X):
        try:
            x_filtered = []
            for x in X:
                if x == x:
                    x_filtered.append(x)
            return x_filtered
        except:
            raise NameError('[Process error] There has been an error while removing empty values.')

    def describe(self, features_names, X):
        """
        returns a Feature object list with each
        mean, std, quartiles, min & max values
        """
        try:
            i = 0
            features = []
            for x in X:
                feature = Feature(features_names[i], self.remove_empty_values(x))
                features.append(feature)
                self.mean.append(feature.mean)
                self.stdev.append(feature.std)
                i +=1
            return features
        except:
            raise NameError('[Process error] There has been an error while processing in describe method.')

    def feature_scale_normalise(self, X):
        """
        Normalises & Standardise each feature vector in X
        using Feature class
        """
        try:
            Xnorm = []
            if not self.mean and not self.stdev:
                features_describe = self.describe(self.features, X)
            for i in range(len(X)):
                Xnorm.append((X[i] - self.mean[i]) / self.stdev[i])
            Xnorm = np.array(Xnorm)
            Xnorm = Xnorm.T
            Xnorm = np.insert(Xnorm, 0, 1, axis=1)
            if self.verbose > 0:
                print('\n-->\tNormalising dataset.')
            return Xnorm
        except NameError as e:
            print(e)
            raise NameError('[Process error] There has been an error while processing (feature scaling/normalising).')

    def one_hot_encoding(self, y):
        """
        Changes each y into 1x4 matrix
        Gryffindor =  [ 1 0 0 0 ]
        Slytherin =  [ 0 1 0 0 ]
        Hufflepuff = [ 0 0 1 0 ]
        Ravenclaw = [ 0 0 0 1 ]
        """
        try:
            y_encoded = []
            m = y.shape[0]
            for i in range(m):
                house_index = self.houses.index(y[i][0])
                y_encoded.append([0, 0, 0, 0])
                y_encoded[i][house_index] = 1
            y_encoded = np.array(y_encoded)
            if self.verbose > 0:
                print('\n-->\tOne hot encodinging the House values.')
            if self.verbose > 1:
                print('\tGryffindor  [ 1 0 0 0 ]')
                print('\tSlytherin   [ 0 1 0 0 ]')
                print('\tHufflepuff  [ 0 0 1 0 ]')
                print('\tRavenclaw   [ 0 0 0 1 ]')
            return y_encoded
        except:
            raise NameError('[Process error] There has been an error while processing (One hot encoding).')

    def pre_activation(self, X, theta):
        """
        Dot product of weights (θ vector) and features (X)
        will be used in activation (sigmoid)
        """
        try:
            ret = np.dot(X, theta)
            return ret
        except:
            raise NameError('[Process error] There has been an error while processing (pre activation).')

    def activation(self, z):
        """
        The activated function is used in the hypothesis.
        Here we use sigmoid function
        """
        try:
            return 1 / (1 + np.exp(-z))
        except:
            raise NameError('[Process error] There has been an error while processing (activation).')

    def hypothesis(self, X, weights):
        """
            Predict the class
            **input: **
                *X: (Numpy Matrix)
                *weights: (Numpy vector)
            **reutrn: (Numpy vector)**
                *0 or 1
        """
        try:
            Z = self.pre_activation(X, weights)
            H = self.activation(Z)
            return H
        except:
            raise NameError('[Process error] There has been an error while processing (hypothesis).')

    def cost(self, H, y, theta):
        """
        Calculates cost for given X and y
        the higher the cost, the more inaccurate the theta values are
        """
        try:
            m = H.shape[0]
            log_not_H = np.log(np.ones(H.shape) - H)
            not_y = np.ones(y.shape) - y
            loss = np.dot(y.T, np.log(H)) + np.dot(not_y.T, log_not_H)
            cost_matrix = (-1/m) * loss
            cost = [cost_matrix[0][0], cost_matrix[1][1], cost_matrix[2][2], cost_matrix[3][3]]
            return np.array(cost)
        except:
            raise NameError('[Process error] There has been an error while processing (cost function).')

    def fit(self, X, y, learning_rate, iter, calculate_cost):
        """
        Gradient descent algorithm to update theta values
        """
        try:
            m = X.shape[0]
            cost_history = []
            if self.verbose > 0:
                print('\n-->\tTRAINING THE MODEL')
            for i in range(iter):
                H = self.hypothesis(X, self.thetas)
                loss = (H - y).T
                loss_per_feature = np.dot(loss, X).T
                self.thetas -= learning_rate * (1/m) * loss_per_feature
                if self.verbose > 2 and i % 25 == 0:
                    print('\n\tTETHAS [ iteration {} ]'.format(i))
                    print(' [ Gryffindor  Slytherin   Hufflepuff  Ravenclaw ]')
                    print('{}'.format(self.thetas))
                if calculate_cost:
                    cost_history.append(self.cost(H, y, self.thetas))
            if self.verbose > 1:
                print('\n\tFINAL TETHAS [ iteration {} ]'.format(i))
                print(' [ Gryffindor  Slytherin   Hufflepuff  Ravenclaw ]')
                print('{}'.format(self.thetas))
            if calculate_cost:
                self.cost_history = cost_history
        except:
            raise NameError('[Process error] There has been an error while processing (Fit function).')

    def validate(self, result, y):
        """
        Compares X_test predictions with actual result
        """
        # try:
        incorrect = 0
        m = result.shape[0]
        print(m)
        true_positive = np.zeros((4, 1))
        false_negative = np.zeros((4, 1))
        false_positive = np.zeros((4, 1))
        for i in range(m):
            pos = np.argmax(y[i])
            house = self.houses[pos]
            if result[i] == house:
                true_positive[pos] += 1
            else:
                false_negative[pos] += 1
                false_positive[self.houses.index(result[i])] += 1
                print('i = {} ||  actual = {} || predicted = {} || pos = {}'.format(i, house, result[i], pos))
        true_negative = (np.ones((4, 1)) * m) - (true_positive + false_negative)
        sensitivity = true_positive / (true_positive + false_negative)
        specificity = true_negative / (true_negative + false_positive)
        precision = true_positive / (true_positive + false_positive)
        accuracy = (true_positive + true_negative) / (true_positive + true_negative + false_positive + false_negative)
        balanced_accuracy = (sensitivity + specificity) / 2
        F1_score = (2 * precision * sensitivity) / (precision + sensitivity)
        print(self.houses)
        print('true_positive')
        print(true_positive.T)
        print('true_negative')
        print(true_negative.T)
        print('false_negative')
        print(false_negative.T)
        print('false_positive')
        print(false_positive.T)

        print('sensitivity')
        print(sensitivity)
        print('specificity')
        print(specificity)
        print('precision')
        print(precision)
        print('accuracy')
        print(accuracy)
        print('balanced_accuracy')
        print(balanced_accuracy)
        print('F1_score')
        print(F1_score)
            # print("Test results : {}".format((m - incorrect)/m))
        # except:
        #     raise NameError('[Process error] There has been an error while processing (validation function).')

    def plot_cost(self):
        print("EMILIE I LOVE U")

    def predict(self, X):
        """
        Predict house name of student given his grades already normalized
        """
        try:
            prediction = self.hypothesis(X, self.thetas)
            result = []
            index = []
            for i in range(prediction.shape[0]):
                house_index = np.argmax(prediction[i])
                result.append(self.houses[house_index])
            result = np.array(result)
            df = pd.DataFrame(data=result, columns=["Hogwarts House"])
            df.index.name = "Index"
            df.to_csv('datasets/houses.csv', index=True)
            return result
        except:
            raise NameError('[Process error] There has been an error while processing (predictor).')
    