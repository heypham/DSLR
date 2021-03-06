#/usr/bin/python3

try:
    import argparse
    import pickle
    from classes.logisticregression import LogisticRegression
except NameError as e:
    print(e)
    print('[Import error] Please run <pip install -r requirements.txt>')
    exit()

def parse_arguments():
    try:
        parser = argparse.ArgumentParser(prog='logreg_train.py', usage='%(prog)s [-h][-v {1,2,3}][-al][-it][-cst] datafile.csv', description='Train the model to predict the Hogwarts house.')
        parser.add_argument('datafile', help='.csv file containing the data to train the model')
        parser.add_argument('-v', '--verbose', help='increase output verbosity', type=int, default=0)
        parser.add_argument('-lr', '--learning_rate', help='[default = 0.01]', type=float, default=0.01)
        parser.add_argument('-it', '--iterations', help='[default = 100]', type=int, default=100)
        parser.add_argument('-tr', '--training_percentage', help='percentage of the dataset to generate the train dataset [default = 0.8]', type=float, default=0.8)
        parser.add_argument('-cst', '--cost', help='cost function', action='store_true')
        parser.add_argument('-f', '--choose_features', help='train logistic model with chosen features', action='store_true')
        parser.add_argument('-ev', '--evaluate', help='display confusion matrix', action='store_true')
        args = parser.parse_args()
        return args
    except:
        raise NameError('\n[Input error]\nThere has been an error while parsing the arguments.\n')

def save_model(model, verbose):
    try:
        outfile = open('logreg_model.42', 'wb')
        pickle.dump(model, outfile)
        outfile.close()
        if verbose > 0:
            print('\n-->\tSaving model')
    except:
        raise NameError('\n[Save error]\nThere has been an error while saving the information.\n')

def main():
    try:
        args = parse_arguments()
        model = LogisticRegression()
        model.set_verbose(args.verbose)
        if args.verbose > 0:
            print('\n[ Process information ]')
        X, y, features_names = model.read_csv(args.datafile, args.choose_features)
        X_clean, y_clean = model.clean_data(X.T, y)
        X_norm = model.feature_scale_normalise(X_clean)
        y_encoded = model.one_hot_encoding(y_clean)
        X_train, X_test, y_train, y_test = model.split_data(X_norm, y_encoded, args.training_percentage)
        model.fit(X_train, y_train, args.learning_rate, args.iterations, args.cost)
        if args.cost:
            model.plot_cost()
        save_model(model, args.verbose)
        if args.verbose > 0:
            print('\n[ Process completed ]\n')
        if args.evaluate:
            model.validate(model.predict(X_test), y_test)
    except NameError as e:
        print(e)

if __name__ == '__main__':
    main()