#/usr/bin/python3
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
                # Change L/R hand to 0/1 values to add them as feature
                if key == 'Best Hand':
                    i = 0
                    val = []
                    for v in value:
                        if v == 'Left':
                            val.append(0)
                        else:
                            val.append(1)
                        i +=1
                    X.append(val)
                else:
                    X.append(value)

        # Transform arrays as numpy arrays for calculations
        X = np.array(X)
        y = np.array(y)
        features = np.array(features).T
        return X, y, features
    except:
        raise NameError('[Read error] Wrong file format. Make sure you give an existing .csv file as argument.')

def clasify_data_per_house(X, y):
    i = 0
    x_filtered_1 = []
    x_filtered_2 = []
    x_filtered_3 = []
    x_filtered_4 = []
    for x in X:
        if x == x:
            if y[i] == 'Gryffindor':
                x_filtered_1.append(x)
            if y[i] == 'Slytherin':
                x_filtered_2.append(x)
            if y[i] == 'Hufflepuff':
                x_filtered_3.append(x)
            if y[i] == 'Ravenclaw':
                x_filtered_4.append(x)
            else:
                pass
        i += 1
    return [x_filtered_1, x_filtered_2, x_filtered_3, x_filtered_4]

def remove_empty_values(X):
    x_filtered = []
    for x in X:
        if x == x:
            x_filtered.append(x)
    return x_filtered

def calc_std(data, mean, count):
    sum_squares = 0
    for i in range(len(data)):
        sum_squares += (data[i] - mean) ** 2
    std = sum_squares / (count - 1)
    std = std ** 0.5
    return std

def find_most_homogeneus_feature(X, y):
    i = 0
    final_i = 0
    std_min = float('inf')
    for x in X:
        data = remove_empty_values(x)
        count = len(data)
        mean = sum(data) / count
        std = calc_std(data, mean, count)
        if std < std_min:
            final_i = i
            std_min = std
        i += 1
    return final_i

def main():
    try:
        args = parse_arg()
        X, y, features_names = read_csv(args.datafile)
        feature_to_plot = find_most_homogeneus_feature(X, y)
        data = clasify_data_per_house(X[feature_to_plot], y[0])
        name = features_names[feature_to_plot]
        bins = 15
        plt.title(name)
        plt.hist(data[0], bins, alpha=0.5, histtype='step', linewidth=2, label='Gry')
        plt.hist(data[1], bins, alpha=0.5, histtype='step', linewidth=2, label='Sly')
        plt.hist(data[2], bins, alpha=0.5, histtype='step', linewidth=2, label='Huf')
        plt.hist(data[3], bins, alpha=0.5, histtype='step', linewidth=2, label='Rav')
        plt.legend()
        plt.show()
    except NameError as e:
        print(e)

if __name__ == '__main__':
    main()
