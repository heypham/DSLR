#/usr/bin/python3
try:
    from classes.logisticregression import LogisticRegression
    from classes.feature import Feature
    import argparse
except NameError as e:
    print(e)
    print('[Import error] Please run <pip install -r requirements.txt>')
    exit()

def parse_arg():
    try:
        parser = argparse.ArgumentParser(prog='describe', usage='%(prog)s [-h] datafile.csv', description='Program describing the dataset given.')
        parser.add_argument('datafile', help='the .csv file containing the dataset')
        args = parser.parse_args()
        return args
    except:
        raise NameError('[Parse error] There has been an error while parsing the arguments.')

def display(features):
    i = 0
    information = ['name', 'count', 'mean', 'std', 'min', 'q_25', 'q_50', 'q_75', 'max']
    print
    for info in information:
        to_print = '{:<7s}'.format(info)
        if info == 'name':
            to_print = ' ' * 7
        for feature in features:
            to_print += feature.get(info)
        print(to_print)
        i += 1
    print

def main():
    try:
        model = LogisticRegression()
        args = parse_arg()
        X, y, features_names = model.read_csv(args.datafile)
        features = model.describe(features_names, X)
        display(features)

        # Real describe function as reference
        # f = pd.read_csv(args.datafile)
        # sum = f.describe()
        # print(sum)
    except NameError as e:
        print(e)

if __name__ == '__main__':
    main()
