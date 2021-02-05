#/usr/bin/python3
try:
    import argparse
    import pandas as pd
    import numpy as np
except:
    print('[Import error] Please run <pip install --user pandas>')
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

def calc_count(X):
    count = 0
    for x in X:
        if x == x:
            count += 1
    return count

def calc_mean(X):
    mean = 0
    total = 0
    for i in range(len(X)):
        if X[i] == X[i]:
            mean += X[i]
            total += 1
    mean /= total
    return mean

def calc_std(X, mean):
    total = len(X)
    sum_squares = 0
    for i in range(total):
        if X[i] == X[i]:
            sum_squares += (X[i] - mean) ** 2
    std = sum_squares / (total - 1)
    std = std ** 0.5
    return std

def calc_min(X):
    min = float("inf")
    for x in X:
        if x == x and x < min:
            min = x
    return min

def calc_max(X):
    max = float("-inf")
    for x in X:
        if x == x and x > max:
            max = x
    return max

# def calc_quartiles(X):


def describe(features, X):
    i = 0
    for x in X:
        print(features[i])
        # print(x)
        count = calc_count(x)
        mean = calc_mean(x)
        std = calc_std(x, mean)
        min = calc_min(x)
        max = calc_max(x)
        print("count : {:.6f}".format(count))
        print("mean  : {:.6f}".format(mean))
        print("std   : {:.6f}".format(std))
        print("min   : {:.6f}".format(min))
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
