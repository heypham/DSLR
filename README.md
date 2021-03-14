# WELCOME TO DSLR

## Summary
42 PROJECT : Algorithm re-constituting Harry Potter's Poudlard’s Sorting Hat using Logistic Regression.  
The model is a logistic regression trained following the gradient descent algorithm

## How to  

### Set up  
```
python3 -m venv venv
venv/bin/pip3 install -r requirements.txt
source venv/bin/activate
```

### Train model  

Run `python3 logreg_train.py datasets/dataset_train.csv`  
This will generate a `logreg_model.42` file which contains the model and its weights.  
_Mandatory argument:_ Dataset file  
_Optional arguments:_ 
- -h: Display help information
- -v {1, 2, 3}: verbosity level
- -lr {float value}: learning rate value (default 0.01
- -it {int value}: number of iterations to train the model (default 100)
- -tr {float value 0.0-1.0}: defines portion of training set when splitting data into training/testing sets
- -cst: plot of the cost function history
- -ev: display evaluation metrics (accuracy, precision, ...)
- -f: asks user to choose which features to use to train the model.

### Predict Houses  

Run `python3 logreg_predict.py datasets/dataset_test.csv`  
This will generate a `houses.csv`file in the `datasets/`folder containing all the results of the model prediction.  
