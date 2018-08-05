import pandas as pd
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor

path = '../input/house-prices-advanced-regression-techniques/train.csv'
# read the data and store data in DataFrame titled melbourne_data
data = pd.read_csv(path) 
testing_data = pd.read_csv('../input/house-prices-advanced-regression-techniques/test.csv')
# print a summary of the data in Melbourne data
print(data.describe())

#  ==================================================================
#  Simple version: use a few features which all have column.dtype=int.
#  ==================================================================
model = DecisionTreeRegressor()

# 1. use a few features
predicators = ["LotArea", "YearBuilt", "1stFlrSF", "2ndFlrSF", "FullBath", "BedroomAbvGr", "TotRmsAbvGrd"]
selected_columns_X = data[predicators]
y = data['SalePrice']
model.fit(selected_columns_X, y)
print('training data set error:', 
      mean_absolute_error(model.predict(selected_columns_X), y))

# Use validation data set to measure training error.
train_x, validation_x, train_y, validation_y = train_test_split(selected_columns_X, 
                                                                y, 
                                                                test_size=0.3, 
                                                                random_state=0)
model.fit(train_x, train_y)
print("validation data set error", 
      mean_absolute_error(model.predict(validation_x), validation_y))
      
      
#  ==================================================================
#  Advanced version: use all features.
#  ==================================================================      
model = DecisionTreeRegressor()

# Y = weight * X + b.  Y = data['SalePrice'].
# ==========================================
# Data cleanup step 1 (cleanup): 
# remove columns with missing values. Also remove the column that we will 
# predict (SalePrice) and the ID.
cols_with_missing_val = [col for col in data.columns 
                         if data[col].isnull().any()]
training_data_without_missing_cols = data.drop(
    columns=['Id','SalePrice'] + cols_with_missing_val)
testing_data_without_missing_cols = testing_data.drop(
    columns=['Id'] + cols_with_missing_val)

# Data cleanup step 2 (normalization): 
# normalize data through get_dummies
training_normalized_X = pd.get_dummies(training_data_without_missing_cols)
testing_normalized_X = pd.get_dummies(testing_data_without_missing_cols)

# Data cleanup step 3 (alignment):
# Make sure training and testing data set have same columns in the same order.
# Use align() function.
final_training_X, final_testing_X = training_normalized_X.align(
    testing_normalized_X, join='left', axis=1)
print("training set column count: ", len(final_training_X.columns), ", testing set: ", len(final_testing_X.columns))

# Model training step 1 
# Use training data to train and calculate error.
y = data['SalePrice']
model.fit(final_training_X, y)
print('training error:', 
      mean_absolute_error(model.predict(final_training_X), y))

# Model training step 2 
# Use training data to train, then use validation data set to calculate error.

# First split training set into training + validation.
# 30% validation, 70% training.
train_x, validation_x, train_y, validation_y = train_test_split(
    final_training_X, y, test_size=0.3, random_state=0)
model.fit(train_x, train_y)
print('validation data set error', 
      mean_absolute_error(model.predict(validation_x), validation_y))
