#/usr/bin/python3

try:
    from classes.logisticregression import LogisticRegression
    import matplotlib.pyplot as plt
except NameError as e:
    print(e)
    print('[Import error] Please run <pip install -r requirements.txt>')
    exit()

def main():
    try:
        model = LogisticRegression()
        args = model.parse_arg()
        X, y, features = model.read_csv(args.datafile)
        X_norm = model.feature_scale_normalise(X)
        X_train, X_test, y_train, y_test = model.split_data(X_norm.T, y)
        y_train_encoded = model.one_hot_encoding(y_train)
        y_test_encoded = model.one_hot_encoding(y_test)

    except NameError as e:
        print(e)

if __name__ == '__main__':
    main()