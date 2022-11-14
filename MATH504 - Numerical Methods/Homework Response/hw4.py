import pandas as pd
from sklearn import preprocessing, svm
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# read in the data
economic_data = pd.read_csv(
    filepath_or_buffer="/Users/NathanBick/Documents/Graduate School/MATH504 - Numerical Methods/Homework/HW4/economic_data.txt",
    sep = "\t",
    skiprows=34)

# split the data
X_train, X_test, y_train, y_test = train_test_split(economic_data, economic_data[], test_size= 1 / 3)



