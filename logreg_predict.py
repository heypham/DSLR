#/usr/bin/python3

try:
    from classes.logisticregression import LogisticRegression
    import pickle
    import argparse
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
        parser.add_argument('-it', '--iterations', help='[default = 1000]', type=int, default=1000)
        parser.add_argument('-tr', '--training_percentage', help='percentage of the dataset to generate the train dataset [default = 0.8]', type=float, default=0.8)
        parser.add_argument('-cst', '--cost', help='cost function', action='store_true')
        args = parser.parse_args()
        return args
    except:
        raise NameError('\n[Input error]\nThere has been an error while parsing the arguments.\n')

def main():
    try:
        saved_model = open("logreg_model.42", "rb")
        model = pickle.load(saved_model)
        saved_model.close()

        args = parse_arguments()
        X, y, features = model.read_csv(args.datafile)
        X_filled = model.fill_data(X, model.mean)
        X_norm = model.feature_scale_normalise(X_filled)
        
        prediction = model.predict(X_norm)
    except NameError as e:
        print(e)

if __name__ == '__main__':
    main()