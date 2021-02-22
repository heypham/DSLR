#/usr/bin/python3
try:
    from classes.logisticregression import LogisticRegression
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
except:
    print('[Import error] Please run <pip install -r requirements.txt>')
    exit()

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
        print(X)
        print(y)
        # print(features.shape)
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

def get_title(feature_name):
    if feature_name == 'Defense Against the Dark Arts':
        return 'Defense'
    if feature_name == 'Care of Magical Creatures':
        return 'Care MC.'
    if feature_name == 'History of Magic':
        return 'Magic H.'
    if feature_name == 'Muggle Studies':
        return 'Muggle S.'
    return feature_name

def format_axis(axs, feature_1, feature_2):
    axs[feature_1, feature_2].set_xticklabels([])
    axs[feature_1, feature_2].set_yticklabels([])
    axs[feature_2, feature_1].set_xticklabels([])
    axs[feature_2, feature_1].set_yticklabels([])
    return axs

def main():
    try:
        model = LogisticRegression()
        args = model.parse_arg()
        X, y, features_names = model.read_csv(args.datafile)
        nb_features = 13
        houses = ['Gryffindor', 'Slytherin', 'Hufflepuff', 'Ravenclaw']
        fig, axs = plt.subplots(nb_features, nb_features)
        for feature_1 in range(nb_features):
            axs[0, feature_1].set_title(get_title(features_names[feature_1]))
            plt.setp(axs[feature_1, 0], ylabel='{}'.format(get_title(features_names[feature_1])))
            for feature_2 in range(feature_1, nb_features):
                axs = format_axis(axs, feature_1, feature_2)
                data = filter_data(X[feature_1], X[feature_2], y[0])
                for house in houses:
                    [X1_per_house, X2_per_house] = clasify_data_per_house(data, house)
                    if feature_1 == feature_2:
                        axs[feature_1, feature_1].hist(X1_per_house, alpha=0.6)
                    else:
                        axs[feature_1, feature_2].scatter(X2_per_house, X1_per_house, alpha=0.07, marker='.')
                        axs[feature_2, feature_1].scatter(X1_per_house, X2_per_house, alpha=0.07, marker='.')
        plt.suptitle('Pair plot')
        plt.show()
    except NameError as e:
        print(e)

if __name__ == '__main__':
    main()
