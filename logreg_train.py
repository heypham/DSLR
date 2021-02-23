#/usr/bin/python3

try:
    from classes.logisticregression import LogisticRegression
    import pandas as pd
    import matplotlib.pyplot as plt
    from sklearn.model_selection import train_test_split
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
        X_train, X_test, y_train, y_test = model.split_data(X_norm, y)

    except NameError as e:
        print(e)

if __name__ == '__main__':
    main()