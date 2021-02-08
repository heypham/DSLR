#/usr/bin/python3
try:
    import argparse
    import pandas as pd
    import numpy as np
    from classes.feature import Feature
except:
    print('[Import error] Please run <pip install -r requirements.txt>')
    exit()

def parse_arg():
    parser = argparse.ArgumentParser(prog='describe', usage='%(prog)s [-h] datafile.csv', description='Program describing the dataset given.')
    parser.add_argument("datafile", help="the .csv file containing the dataset")
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
            if key == "Index" or key == "First Name" or key == "Last Name" or key == "Birthday":
                pass
            elif key == "Hogwarts House":
                y.append(value)
            else:
                features.append(key)
                # Change L/R hand to 0/1 values to add them as feature
                if key == "Best Hand":
                    i = 0
                    val = []
                    for v in value:
                        if v == "Left":
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

def remove_empty_values(X):
    x_filtered = []
    for x in X:
        if x == x:
            x_filtered.append(x)
    return x_filtered

def describe(features_names, X):
    i = 0
    features = []
    for x in X:
        feature = Feature(features_names[i], remove_empty_values(x))
        features.append(feature)
        i +=1
    return features

def display(features):
    i = 0
    information = ['name', 'count', 'mean', 'std', 'min', 'q_25', 'q_50', 'q_75', 'max']
    for info in information:
        to_print = '{:<7s}'.format(info)
        for feature in features:
            to_print += feature.get(info)
        print(to_print)
        # print('{:15s} {:15s} {:15s} {:15s} {:15s} {:15s} {:15s}'.format(information[i], data[0], data[1], data[2], data[3], data[4], data[5], data[6], data[7]))
        i += 1

def main():
    try:
        args = parse_arg()
        X, y, features_names = read_csv(args.datafile)
        features = describe(features_names, X)
        display(features)

        # Real describe function as reference
        f = pd.read_csv(args.datafile)
        sum = f.describe()
        print(sum)
    except NameError as e:
        print(e)

if __name__ == "__main__":
    main()
