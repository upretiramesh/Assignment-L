import numpy as np

from ACIT4630.polynomialFeatures import PolynomialFeature
from ACIT4630.normalization import Normalization
import ACIT4630.accuracyEvaluator as accuracy
from ACIT4630.regression import LinearRegression

from sklearn.datasets import load_boston

# load the features data
boston = load_boston()
data = boston.data

# load the target data
y = boston.target
y = np.expand_dims(y, axis=1)
print('Before removing outlier: Shape: ', data.shape, y.shape)

# Removing outlier from the dataset

# remove crime greater than 30
idx = np.where(data[:, 0] < 30)
data = data[idx]
y = y[idx]


# remove weighted mean of distances to five Boston employment centres greater than 10
idx = np.where(data[:, 7] < 10)
data = data[idx]
y = y[idx]

idx = np.where(data[:, -1] < 34)
data = data[idx]
y = y[idx]
print('After removing outlier: ', data.shape, y.shape)

# total number of records
num = len(data)

# apply normalization to the data
norm = Normalization()
data_nor = norm.apply(data)
print('Normalization is completed')

# apply PolynomialFeature :default is degree 2
# poly = PolynomialFeature()
# data_poly = poly.transform(data=data_nor)

# split data data into train and test set
# X_train, X_test = data_poly[:int(num*0.8), :], data_poly[int(num*0.8):, :]
# y_train, y_test = y[:int(num*0.8), :], y[int(num*0.8):, :]
# print('X_train:', X_train.shape, 'X_test: ', X_test.shape)
# print('Data divided into train and test set')

# Split the data into train and test set
X_train, X_test = data_nor[:int(num * 0.8), :], data_nor[int(num * 0.8):, :]
y_train, y_test = y[:int(num * 0.8), :], y[int(num * 0.8):, :]
print('X_train:', X_train.shape, 'X_test: ', X_test.shape)
print('Data divided into train and test set')

# Define linear regression
model = LinearRegression()
model.fit(X_train, y_train)
print('Model training is complete')

# predict the response value
yhat = model.predict(X_test)
print('Prediction of response value is completed')

print('#######################')
# print(y_test[:20])
# print(yhat[:20])
print('######################')

# Calculate accuracy
print('Mean square error: ', accuracy.mse_score(y_test, yhat))
print('Root mean square error: ', accuracy.rmse_score(y_test, yhat))
print('Residual sum of square:', accuracy.rss_score(y_test, yhat))
print('Mean absolute error: ', accuracy.mae_score(y_test, yhat))
print('R-square error: ', accuracy.r2_score(y_test, yhat))
