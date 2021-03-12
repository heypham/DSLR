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
        parser = argparse.ArgumentParser(prog='logreg_predict.py', usage='%(prog)s [-h][-v {1,2,3}][-al][-it][-cst] datafile.csv', description='Train the model to predict the Hogwarts house.')
        parser.add_argument('datafile', help='.csv file containing the data to train the model')
        parser.add_argument('-v', '--verbose', help='increase output verbosity', type=int, default=0)
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
        X, y, features_names = model.read_csv(args.datafile)
        X_filled = model.fill_data(X, model.mean)
        X_norm = model.feature_scale_normalise(X_filled)
        
        prediction = model.predict(X_norm)
    except NameError as e:
        print(e)

if __name__ == '__main__':
    main()