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
        # print(y_clean[0])
        y_encoded = model.one_hot_encoding2(y_clean)
        y_griffindor = y_encoded[:,0].reshape(-1, 1)
        for i in range(100):
            H = model.hypothesis(X_norm, model.theta)
            model.theta -= ((np.dot((H - y_griffindor).T, X_norm)) / H.shape[0]).T
        # print(model.theta)
        # print(model.theta.shape)
        H_decoded = []
        for i in range(H.shape[0]):
            if H[i][0] > 0.5:
                H_decoded.append(1)
            else:
                H_decoded.append(0)
        H_decoded = np.array(H_decoded).reshape(-1, 1)

        for i in range(H_decoded.shape[0]):
            if H_decoded[i][0] != y_griffindor[i][0]:
                print('upsi')

        # print(H_decoded)
        # print(y_griffindor)
        # print(H_decoded.shape)
    except NameError as e:
        print(e)

if __name__ == '__main__':
    main()