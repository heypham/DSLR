#/usr/bin/python3
try:
    from classes.logisticregression import LogisticRegression
    import matplotlib.pyplot as plt
    import argparse
except:
    print('[Import error] Please run <pip install -r requirements.txt>')
    exit()

def parse_arg():
    try:
        parser = argparse.ArgumentParser(prog='histogram', usage='%(prog)s [-h] datafile.csv', description='Program showing which course has a homogeneous score distribution between all four houses.')
        parser.add_argument('datafile', help='the .csv file containing the dataset')
        parser.add_argument('-a', '--all', help='plot all the features', action='store_true')
        args = parser.parse_args()
        return args
    except:
        raise NameError('[Parse error] There has been an error while parsing the arguments.')

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

def plot(data, name):
        bins = 15
        plt.suptitle("Which course has a homogeneous score distribution between all four houses?")
        plt.title(name)
        plt.hist(data[0], bins, alpha=0.5, histtype='step', linewidth=2, label='Gry')
        plt.hist(data[1], bins, alpha=0.5, histtype='step', linewidth=2, label='Sly')
        plt.hist(data[2], bins, alpha=0.5, histtype='step', linewidth=2, label='Huf')
        plt.hist(data[3], bins, alpha=0.5, histtype='step', linewidth=2, label='Rav')
        plt.legend()
        plt.show()

def main():
    try:
        model = LogisticRegression()
        args = parse_arg()
        X, y, features_names = model.read_csv(args.datafile)
        if (args.all)
            for i in features_name.len():
                data = clasify_data_per_house(X[i], y[0])
                name = features_names[i]
                plot(data, name)
        else
            feature_to_plot = find_most_homogeneus_feature(X, y)
            data = clasify_data_per_house(X[feature_to_plot], y[0])
            name = features_names[feature_to_plot]
            plot(data, name)
    except NameError as e:
        print(e)

if __name__ == '__main__':
    main()
