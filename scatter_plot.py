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
        x1_per_house = []
        x2_per_house = []
        for i in range(len(data[0])):
            if data[2][i] == house:
                x1_per_house.append(data[0][i])
                x2_per_house.append(data[1][i])
            i += 1
        return [x1_per_house, x2_per_house]
    except:
        return [[], []]

def calc_cov(X_1, X_2, y):
    cov = 0
    data = filter_data(X_1, X_2, y)
    X1 = data[0]
    X2 = data[1]
    mean_x1 = sum(X1) / len(X1)
    mean_x2 = sum(X2) / len(X2)
    for i in range(len(X1)):
        cov += (X1[i] - mean_x1) * (X2[i] - mean_x2)
        i += 1
    cov /= (len(X1) - 1)
    print(cov)
    return cov

def find_most_correlated_features(X, y):
    i = 0
    j = 1
    final_i = 0
    final_j = 0
    final_cov = 0
    for i in range(13):
        for j in range(i + 1, 13):
            cov = abs(calc_cov(X[i], X[j], y))
            if final_cov < cov:
                final_i = i
                final_j = j
                final_cov = cov
            print('i: {} || j: {}'.format(i, j))
            j += 1
        i += 1
    print(final_i)
    print(final_j)
    return [final_i, final_j]

def main():
    try:
        args = parse_arg()
        X, y, features_names = read_csv(args.datafile)
        features_to_plot = find_most_correlated_features(X, y[0])
        print(features_names[features_to_plot[0]])
        print(features_names[features_to_plot[1]])
        data = filter_data(X[features_to_plot[0]], X[features_to_plot[1]], y[0])
        # name = features_names[feature_to_plot]
        houses = ['Gryffindor', 'Slytherin', 'Hufflepuff', 'Ravenclaw']
        for house in houses:
            data_per_house = clasify_data_per_house(data, house)
            plt.scatter(data_per_house[0], data_per_house[1], alpha=0.2, label=house)
        plt.legend()
        plt.show()
        exit()
    except NameError as e:
        print(e)

if __name__ == '__main__':
    main()
