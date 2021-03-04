#/usr/bin/python3

try:
    from classes.logisticregression import LogisticRegression
    import matplotlib.pyplot as plt
    import numpy as np
except NameError as e:
    print(e)
    print('[Import error] Please run <pip install -r requirements.txt>')
    exit()

def main():
    try:
        model = LogisticRegression()
        args = model.parse_arg()
        X, y, features = model.read_csv(args.datafile)
        X_clean, y_clean = model.clean_data(X.T, y)
        X_norm = model.feature_scale_normalise(X_clean)

        # X_train, X_test, y_train, y_test = model.split_data(X_norm.T, y_clean)
        # y_train_encoded = model.one_hot_encoding(y_train)
        # y_test_encoded = model.one_hot_encoding(y_test)

        y_encoded = model.one_hot_encoding(y_clean)
        tethas = model.fit(X_norm, y_encoded, 0.1, 100)
        H = model.H_from_probability_to_absolute_values(X_norm)
        for i in range(H.shape[0]):
            if (H[i] != y_encoded[i]).all():
                print('H = {} || y = {}'.format(H[i], y_encoded[i]))

    except NameError as e:
        print(e)

if __name__ == '__main__':
    main()