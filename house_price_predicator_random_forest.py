import pandas as pd

path = '../input/house-prices-advanced-regression-techniques/train.csv'
# read the data and store data in DataFrame titled melbourne_data
data = pd.read_csv(path) 
testing_data = pd.read_csv('../input/house-prices-advanced-regression-techniques/test.csv')

def predict(model, training_x, training_y, validation_x, validation_y, max_leaf_nodes=None):
    model.fit(training_x, training_y)
    print('max_leaf_nodes:',  max_leaf_nodes, ', error:', 
          mean_absolute_error(
              model.predict(validation_x), validation_y))

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

# Y = weight * X + b.  Y = data['SalePrice'].
# =============Cleanup=============================
# Data cleanup step 1 (cleanup): 
# remove columns with missing values. Also remove the column that we will 
# predict (SalePrice) and the ID.
cols_with_missing_val = [col for col in data.columns 
                         if data[col].isnull().any()]
training_data_without_missing_cols = data.drop(
    columns=['Id','SalePrice'] + cols_with_missing_val)
testing_data_without_missing_cols = testing_data.drop(
    columns=['Id'] + [col for col in testing_data.columns
                     if testing_data[col].isnull().any()])

# =============Cleanup=============================
# Data cleanup step 2 (normalization): 
# normalize data through get_dummies
training_normalized_X = pd.get_dummies(training_data_without_missing_cols)
testing_normalized_X = pd.get_dummies(testing_data_without_missing_cols)


# =============Cleanup=============================
# Data cleanup step 3 (alignment):
# Make sure training and testing data set have same columns in the same order.
# Use align() function.
final_training_X, final_testing_X = training_normalized_X.align(
    testing_normalized_X, join='inner', axis=1)
print("training set column count: ", len(final_training_X.columns), ", testing set: ", len(final_testing_X.columns))

# ==============Model Training============================
# Model training step 1 
# Use training data to train and calculate error.
y = data['SalePrice']

# ==============Model Training============================
# Model training step 2 
# Use training data to train, then use validation data set to calculate error.

# First split training set into training + validation.
from sklearn.model_selection import train_test_split
# 30% validation, 70% training.
train_x, validation_x, train_y, validation_y = train_test_split(
    final_training_X, y, test_size=0.3, random_state=0)

for nodes in (500,1000,5000,7000):
    model = RandomForestRegressor(max_leaf_nodes=nodes)
    print('training set error')    
    predict(model, final_training_X, y, final_training_X, y, 
            max_leaf_nodes=nodes)
    print('validation set error')
    predict(model, train_x, train_y, validation_x, validation_y,
           max_leaf_nodes=nodes)

print('---no max leaf nodes---')
predict(model, final_training_X, y, final_training_X, y, 
            max_leaf_nodes=None)
predict(model, train_x, train_y, validation_x, validation_y,
           max_leaf_nodes=None)


# =======Plot=========
import matplotlib.pyplot as plt
fig = plt.figure()
ax1 = fig.add_subplot(111)
ax1.scatter(model.predict(train_x), train_y, s=1, c='b', marker="s", label='real')
ax1.scatter(model.predict(validation_x),validation_y, s=10, c='r', marker="o")
plt.show()

# ==============Run Model With Testing Data===========
model = RandomForestRegressor(max_leaf_nodes=5000)
model.fit(train_x, train_y)
predicted_prices = model.predict(final_testing_X)
print(predicted_prices)

# ==============Submit to kaggle===========
my_submission = pd.DataFrame({'Id': testing_data.Id, 'SalePrice': predicted_prices})
# you could use any filename. We choose submission here
my_submission.to_csv('submission.csv', index=False)
