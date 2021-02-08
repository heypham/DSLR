#/usr/bin/python3
try:
    import argparse
    import pandas as pd
    import numpy as np
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

def filter_dataset(X):
    x_filtered = []
    for x in X:
        if x == x:
            x_filtered.append(x)
    return x_filtered

def calc_count(X):
    return len(X)

def calc_mean(X, count):
    return sum(X) / count

def calc_std(X, mean, count):
    sum_squares = 0
    for i in range(len(X)):
        sum_squares += (X[i] - mean) ** 2
    std = sum_squares / (count - 1)
    std = std ** 0.5
    return std

def calc_quartiles(X, quartile, count):
    X.sort()
    position_min = (float(quartile) / 100) * (count - 1)
    position_max_coef = position_min - int(position_min)
    if position_max_coef == 0.0:
        return X[int(position_min)]
    position_max = position_min + 1
    position_min_coef = 1 - position_max_coef
    result_min = (X[int(position_min)] * position_min_coef)
    result_max = (X[int(position_max)] * position_max_coef)
    return result_min + result_max

def describe(features, X):
    i = 0
    for x in X:
        print(features[i])
        x_filtered = filter_dataset(x)
        count = calc_count(x_filtered)
        mean = calc_mean(x_filtered, count)
        std = calc_std(x_filtered, mean, count)
        min = calc_quartiles(x_filtered, 0, count)
        quartile_25 = calc_quartiles(x_filtered, 25, count)
        quartile_50 = calc_quartiles(x_filtered, 50, count)
        quartile_75 = calc_quartiles(x_filtered, 75, count)
        max = calc_quartiles(x_filtered, 100, count)
        print("count : {:.6f}".format(count))
        print("mean  : {:.6f}".format(mean))
        print("std   : {:.6f}".format(std))
        print("min   : {:.6f}".format(min))
        print("25%   : {:.6f}".format(quartile_25))
        print("50%   : {:.6f}".format(quartile_50))
        print("75%   : {:.6f}".format(quartile_75))
        print("max   : {:.6f}\n".format(max))
        i +=1

def main():
    try:
        args = parse_arg()
        X, y, features = read_csv(args.datafile)
        # print(features)
        describe(features, X)

        # Real describe function as reference
        f = pd.read_csv(args.datafile)
        sum = f.describe()
        print(sum)
    except NameError as e:
        print(e)

if __name__ == "__main__":
    main()
