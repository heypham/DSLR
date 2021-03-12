#/usr/bin/python3

try:
    import numpy as np
    import pandas as pd
    from classes.feature import Feature
    import matplotlib.pyplot as plt
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
        self.init_thetas = True
        self.cost_history = []
        self.dropped_features = []
        self.choose_features = None

    def read_csv(self, datafile, choose_feature=False):
        """
        Function to read csv file and split into training/testing sets
        returns train/test sets for X and y + list of 4 houses
        """
        try:
            if self.choose_features is None:
                self.choose_features = choose_feature
            f = pd.read_csv(datafile)
            houses = f['Hogwarts House'].unique()
            features = list(f.columns[6:])
            self.features = np.array(features).T
            features_to_drop = []
            y = f['Hogwarts House']
            unused_features = ['Index','Hogwarts House', 'First Name', 'Last Name', 'Birthday', 'Best Hand']
            X = f.drop(unused_features,axis=1)
            if self.choose_features is True:
                self.dropped_features = self.features_to_drop()
                if len(self.dropped_features) == 0:
                    print("An error was detected in the chosen features. Will train model using all available features.")
                else:
                    X = X.drop(self.dropped_features, axis=1)
                    if self.init_thetas == True:
                        self.thetas = np.zeros((14 - len(self.dropped_features), 4))
                        self.init_thetas = False
            self.X = np.array(X).T
            self.y = np.array([y])
            if self.verbose > 0:
                print('\n-->\tReading CSV file.')
            return self.X, self.y, self.features
        except NameError as e:
            print(e)
            raise NameError('[Read error] Wrong file format. Make sure you give an existing .csv file as argument.')

    def features_to_drop(self):
        if len(self.dropped_features) > 0:
            return self.dropped_features
        else:
            print("Here is the list of features.\n")
            for i in range(len(self.features)):
                print("{}. {}".format(i+1, self.features[i]))
            print("\nPlease enter the feature numbers you want to use to train the model (separated by space).")
            print("Example : 1 6 4\n")
            training_features = input()
            print()
            training_features = training_features.split(' ')
            error = 0
            for i in range(len(training_features)):
                try:
                    training_features[i] = self.features[int(training_features[i]) - 1]
                except:
                    error = 1
            if error == 0:
                dropping_features = list(self.features)
                for chosen_feature in training_features:
                    for feature in dropping_features:
                        if feature == chosen_feature:
                            dropping_features.remove(feature)
                print("Training logistic regression model with the following features\n{}".format(training_features))
            else:
                dropping_features = []
        return dropping_features

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
        except NameError as e:
            print(e)
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

    def describe(self, feature_names, X):
        """
        returns a Feature object list with each
        mean, std, quartiles, min & max values
        """
        try:
            i = 0
            features = []
            for x in X:
                feature = Feature(feature_names[i], self.remove_empty_values(x))
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
        except NameError as e:
            print(e)
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
                    print('\n\tTethas [ iteration {} ]'.format(i))
                    print(' [ Gryffindor  Slytherin   Hufflepuff  Ravenclaw ]')
                    print('{}'.format(self.thetas))
                if calculate_cost:
                    cost_history.append(self.cost(H, y, self.thetas))
            if self.verbose > 1:
                print('\n\tFinal Tethas [ iteration {} ]\n'.format(i))
                print(' [ Gryffindor  Slytherin   Hufflepuff  Ravenclaw ]\n')
                print('{}'.format(self.thetas))
            if calculate_cost:
                self.cost_history = np.array(cost_history)
        except NameError as e:
            print(e)
            raise NameError('[Process error] There has been an error while processing (Fit function).')

    def validate(self, result, y):
        """
        Compares X_test predictions with actual result
        """
        try:
            m = result.shape[0]
            true_positive = np.zeros((4, 1))
            false_negative = np.zeros((4, 1))
            false_positive = np.zeros((4, 1))
            if self.verbose > 0:
                print('\n[ Model evaluation ]\n')
            if self.verbose > 1:
                print('+-----------------------------------------------+')
                print ('|\t\tPREDICTION ERRORS\t\t|')
                print('+-----------------------+-----------------------+')
                print('|\tReal House\t|\tPredicted\t|')
                print('+-----------------------+-----------------------+')
            for i in range(m):
                pos = np.argmax(y[i])
                predicted_house = self.houses[pos]
                if result[i] == predicted_house:
                    true_positive[pos] += 1
                else:
                    false_negative[pos] += 1
                    false_positive[self.houses.index(result[i])] += 1
                    if self.verbose > 1:
                        print('|\t{}\t|\t{}\t|'.format(predicted_house, result[i]))
            if self.verbose > 1:
                print ('+-----------------------+-----------------------+\n\n')
            true_negative = (np.ones((4, 1)) * m) - (true_positive + false_negative)
            sensitivity = true_positive / (true_positive + false_negative)
            specificity = true_negative / (true_negative + false_positive)
            precision = true_positive / (true_positive + false_positive)
            accuracy = (true_positive + true_negative) / (true_positive + true_negative + false_positive + false_negative)
            balanced_accuracy = (sensitivity + specificity) / 2
            F1_score = (2 * precision * sensitivity) / (precision + sensitivity)
            if self.verbose > 1:
                print('                        +-----------------------+-----------------------+-----------------------+-----------------------+-----------------------+')
                print('                        |\t{}\t|\t{}\t|\t{}\t|\t{}\t|\t{}\t\t|'.format(self.houses[0], self.houses[1], self.houses[2], self.houses[3], 'Mean'))
                print('+-----------------------+-----------------------+-----------------------+-----------------------+-----------------------+-----------------------+')
                print('|\t{}\t|\t{}\t\t|\t{}\t\t|\t{}\t\t|\t{}\t\t|\t{}\t\t|'.format('True Positive', true_positive[0][0], true_positive[1][0], true_positive[2][0], true_positive[3][0], np.mean(true_positive)))
                print('+-----------------------+-----------------------+-----------------------+-----------------------+-----------------------+-----------------------+')
                print('|\t{}\t|\t{}\t\t|\t{}\t\t|\t{}\t\t|\t{}\t\t|\t{}\t\t|'.format('True Negative', true_negative[0][0], true_negative[1][0], true_negative[2][0], true_negative[3][0], np.mean(true_negative)))
                print('+-----------------------+-----------------------+-----------------------+-----------------------+-----------------------+-----------------------+')
                print('|\t{}\t|\t{}\t\t|\t{}\t\t|\t{}\t\t|\t{}\t\t|\t{}\t\t|'.format('False Positive', false_positive[0][0], false_positive[1][0], false_positive[2][0], false_positive[3][0], np.mean(false_positive)))
                print('+-----------------------+-----------------------+-----------------------+-----------------------+-----------------------+-----------------------+')
                print('|\t{}\t|\t{}\t\t|\t{}\t\t|\t{}\t\t|\t{}\t\t|\t{}\t\t|'.format('False Negative', false_negative[0][0], false_negative[1][0], false_negative[2][0], false_negative[3][0], np.mean(false_negative)))
                print('+-----------------------+-----------------------+-----------------------+-----------------------+-----------------------+-----------------------+\n\n')
            print('                        +-----------------------+-----------------------+-----------------------+-----------------------+-----------------------+')
            print('                        |\t{}\t|\t{}\t|\t{}\t|\t{}\t|\t{}\t\t|'.format(self.houses[0], self.houses[1], self.houses[2], self.houses[3], 'Mean'))
            print('+-----------------------+-----------------------+-----------------------+-----------------------+-----------------------+-----------------------+')
            print('|\t{}\t|\t{:.5f}\t\t|\t{:.5f}\t\t|\t{:.5f}\t\t|\t{:.5f}\t\t|\t{:.5f}\t\t|'.format('Sensitivity', sensitivity[0][0], sensitivity[1][0], sensitivity[2][0], sensitivity[3][0], np.mean(sensitivity)))
            print('+-----------------------+-----------------------+-----------------------+-----------------------+-----------------------+-----------------------+')
            print('|\t{}\t|\t{:.5f}\t\t|\t{:.5f}\t\t|\t{:.5f}\t\t|\t{:.5f}\t\t|\t{:.5f}\t\t|'.format('Specificity', specificity[0][0], specificity[1][0], specificity[2][0], specificity[3][0], np.mean(specificity)))
            print('+-----------------------+-----------------------+-----------------------+-----------------------+-----------------------+-----------------------+')
            print('|\t{}\t|\t{:.5f}\t\t|\t{:.5f}\t\t|\t{:.5f}\t\t|\t{:.5f}\t\t|\t{:.5f}\t\t|'.format('Precision', precision[0][0], precision[1][0], precision[2][0], precision[3][0], np.mean(precision)))
            print('+-----------------------+-----------------------+-----------------------+-----------------------+-----------------------+-----------------------+')
            print('|\t{}\t|\t{:.5f}\t\t|\t{:.5f}\t\t|\t{:.5f}\t\t|\t{:.5f}\t\t|\t{:.5f}\t\t|'.format('Accuracy', accuracy[0][0], accuracy[1][0], accuracy[2][0], accuracy[3][0], np.mean(accuracy)))
            print('+-----------------------+-----------------------+-----------------------+-----------------------+-----------------------+-----------------------+')
            print('|\t{}\t|\t{:.5f}\t\t|\t{:.5f}\t\t|\t{:.5f}\t\t|\t{:.5f}\t\t|\t{:.5f}\t\t|'.format('Balanced Acc.', balanced_accuracy[0][0], balanced_accuracy[1][0], balanced_accuracy[2][0], balanced_accuracy[3][0], np.mean(balanced_accuracy)))
            print('+-----------------------+-----------------------+-----------------------+-----------------------+-----------------------+-----------------------+')
            print('|\t{}\t|\t{:.5f}\t\t|\t{:.5f}\t\t|\t{:.5f}\t\t|\t{:.5f}\t\t|\t{:.5f}\t\t|'.format('F1 Score', F1_score[0][0], F1_score[1][0], F1_score[2][0], F1_score[3][0], np.mean(F1_score)))
            print('+-----------------------+-----------------------+-----------------------+-----------------------+-----------------------+-----------------------+\n\n')
            if self.verbose > 1:
                print('Sensitivity       ->\tTP / ( TP + FN )')
                print('Specficity        ->\tTN / ( TN + FP )')
                print('Precision         ->\tTP / ( TP + FP )')
                print('Accuracy          ->\t( TP + TN ) / ( TP + FN + FP + FN )')
                print('Balanced Accuracy ->\t( Sensitivity + Specificity ) / 2')
                print('F1 Score          ->\t( 2 * Precision * Sensitivity ) / ( Precision + Sensitivity )\n')
        except:
            raise NameError('[Process error] There has been an error while processing (validation function).')

    def plot_cost(self):
        plt.title('Cost history plot')
        x = range(len(self.cost_history))
        self.cost_history = self.cost_history.T
        for i in range(len(self.cost_history)):
            plt.plot(x, self.cost_history[i], label=self.houses[i], color=self.colors[self.houses[i]])
        plt.xlabel('iteration')
        plt.ylabel('cost')
        plt.legend()
        plt.show()

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
    