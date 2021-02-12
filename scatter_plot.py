#/usr/bin/python3
try:
    import argparse
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
except:
    print('[Import error] Please run <pip install -r requirements.txt>')
    exit()

def parse_arg():
    parser = argparse.ArgumentParser(prog='describe', usage='%(prog)s [-h] datafile.csv', description='Program describing the dataset given.')
    parser.add_argument('datafile', help='the .csv file containing the dataset')
    args = parser.parse_args()
    return args

def read_csv(datafile):
    try:
        f = pd.read_csv(datafile)
        features = []
        X = []
        y = []
        for key, value in f.iteritems(): 
            # Append features to X matrix
            if key == 'Index' or key == 'First Name' or key == 'Last Name' or key == 'Birthday' or key == 'Best Hand':
                pass
            elif key == 'Hogwarts House':
                y.append(value)
            else:
                features.append(key)
                X.append(value)
        # Transform arrays as numpy arrays for calculations
        X = np.array(X)
        y = np.array(y)
        features = np.array(features).T
        return X, y, features
    except:
        raise NameError('[Read error] Wrong file format. Make sure you give an existing .csv file as argument.')

def filter_data(X1, X2, y):
    try:
        i = 0
        x1_filtered = []
        x2_filtered = []
        y_filtered = []
        for x1 in X1:
            if x1 == x1:
                x2 = X2[i]
                if x2 == x2:
                    x1_filtered.append(x1)
                    x2_filtered.append(x2)
                    y_filtered.append(y[i])
            i += 1
        return [x1_filtered, x2_filtered, y_filtered]
    except:
        return [[], [], []]

def clasify_data_per_house(data, house):
    try:
        i = 0
        count = len(data[0])
        x1_per_house = []
        x2_per_house = []
        for i in range(count):
            if data[2][i] == house:
                x1_per_house.append(data[0][i])
                x2_per_house.append(data[1][i])
            i += 1
        return [x1_per_house, x2_per_house]
    except:
        return [[], []]

def calc_std(X):
    try:
        count = len(X)
        mean = sum(X) / count
        sum_squares = 0
        for i in range(count):
            sum_squares += (X[i] - mean) ** 2
        std = sum_squares / (count - 1)
        std = std ** 0.5
        return std
    except:
        return 0

def calc_pearson_coef(X_1, X_2, y):
    try:
        pearson_coef = 0
        [X1, X2, y] = filter_data(X_1, X_2, y)
        count = len(X1)
        mean_X1 = sum(X1) / count
        mean_X2 = sum(X2) / count
        for i in range(count):
            pearson_coef += ((X1[i] - mean_X1) * (X2[i] - mean_X2)) / (count - 1)
            i += 1
        pearson_coef /= (calc_std(X1) * calc_std(X2))
        return abs(pearson_coef)
    except:
        return 0

def find_most_correlated_features(X, y):
    try:
        feature_1 = 0
        feature_2 = 1
        final_feature_1 = 0
        final_feature_2 = 0
        final_pearson_coef = 0
        for feature_1 in range(13):
            for feature_2 in range(feature_1 + 1, 13):
                pearson_coef = calc_pearson_coef(X[feature_1], X[feature_2], y)
                if final_pearson_coef < pearson_coef:
                    final_feature_1 = feature_1
                    final_feature_2 = feature_2
                    final_pearson_coef = pearson_coef
                feature_2 += 1
            feature_1 += 1
        return final_feature_1, final_feature_2, final_pearson_coef
    except:
        return 0, 0, 'error'

def main():
    try:
        args = parse_arg()
        X, y, features_names = read_csv(args.datafile)
        feature_to_plot_1, feature_to_plot_2, pearson_coef = find_most_correlated_features(X, y[0])
        data = filter_data(X[feature_to_plot_1], X[feature_to_plot_2], y[0])
        houses = ['Gryffindor', 'Slytherin', 'Hufflepuff', 'Ravenclaw']
        plt.suptitle('Scatter plot')
        plt.title('Pearson\'s Coef: {:.3f}'.format(pearson_coef))
        for house in houses:
            [X1, X2] = clasify_data_per_house(data, house)
            plt.scatter(X1, X2, alpha=0.2, label=house)
        plt.xlabel(features_names[feature_to_plot_1])
        plt.ylabel(features_names[feature_to_plot_2])
        plt.legend()
        plt.show()
    except NameError as e:
        print(e)

if __name__ == '__main__':
    main()
