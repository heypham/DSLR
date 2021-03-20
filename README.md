# WELCOME TO DSLR

## Summary
42 PROJECT: Algorithm re-constituting Harry Potter's Poudlard’s Sorting Hat using Logistic Regression.
The model is a logistic regression trained following the gradient descent algorithm.

## How to
### 1. Set up  
```
python3 -m venv venv
venv/bin/pip3 install -r requirements.txt
source venv/bin/activate
```

### 2. Data Analysis
#### 2.1. Describe

Run `python describe.py datasets/dataset_train.csv`

_Mandatory argument:_ Dataset file

--> Displays the count, mean, standard desviation, min, 25% percentil, median, 75% percentil and max value of all the numerical features (except the index) in the dataset pased as an argument.

### 3. Data Visualization
#### 3.1. Histogram

Run `python describe.py datasets/dataset_train.csv`

_Mandatory argument:_ Dataset file  

_Optional arguments:_
- -a: Plot the histogram of all the features

--> Plots the histogram of the course (feature) with the most homogeneous score disribution between all four houses: the course with the smallest standard desviation.

#### 3.2. Scatter plot

Run `python scatter_plot.py datasets/dataset_train.csv`

_Mandatory argument:_ Dataset file  

_Optional arguments:_
- -a: Plot the histogram of all the features

--> Plots the scatter plot of the two features that are more similar. The Peason's coeficient of all the pairs of fetures is calculated to find out the ones that are correlated.

#### 3.3. Pair plot

Run `python pair_plot.py datasets/dataset_train.csv`

_Mandatory argument:_
- Dataset file

--> Plots the scater plot of all the combination of features and the histogram of all of them. From this visualization we can deduce from which features we can extract more informatin to train our model.

### 4. Logistic Regression

#### 4.1. Train model  

Run `python logreg_train.py datasets/dataset_train.csv`

_Mandatory argument:_ Dataset file  

_Optional arguments:_ 
- -h: Display help information
- -v {1, 2, 3}: verbosity level
- -lr {float value}: learning rate value (default 0.01)
- -it {int value}: number of iterations to train the model (default 100)
- -tr {float value 0.0-1.0}: defines portion of training set when splitting data into training/testing sets
- -cst: plot of the cost function history
- -ev: display evaluation metrics (accuracy, precision, ...)
- -f: asks user to choose which features to use to train the model

--> Generates a `logreg_model.42` file which contains the model and its weights.

### 4.2. Predict Houses  

Run `python3 logreg_predict.py logreg_model.42 datasets/dataset_test.csv`

_Mandatory argument:_
- Dataset file
- Model and weights

--> This will generate a `houses.csv` file in the `datasets/` folder containing all the results of the model cointained in `logreg_model.42` prediction for the `dataset_train.csv`.
