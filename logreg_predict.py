#/usr/bin/python3

try:
    from classes.logisticregression import LogisticRegression
    import pickle
except NameError as e:
    print(e)
    print('[Import error] Please run <pip install -r requirements.txt>')
    exit()

def main():
    try:
        saved_model = open("logreg_model.42", "rb")
        model = pickle.load(saved_model)
        saved_model.close()

        args = model.parse_arg()
        X, y, features = model.read_csv(args.datafile)
        X_filled = model.fill_data(X, model.mean)
        X_norm = model.feature_scale_normalise(X_filled)
        
        prediction = model.predict(X_norm)
    except NameError as e:
        print(e)

if __name__ == '__main__':
    main()