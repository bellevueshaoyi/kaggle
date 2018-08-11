import pandas as pd

path = '../input/house-prices-advanced-regression-techniques/train.csv'
# read the data and store data in DataFrame titled melbourne_data
data = pd.read_csv(path) 
testing_data = pd.read_csv('../input/house-prices-advanced-regression-techniques/test.csv')
print(data.describe())

def predict(model, training_x, training_y, validation_x, validation_y, max_leaf_nodes=None):
    model.fit(training_x, training_y)
    print('error:', 
          mean_absolute_error(
              model.predict(validation_x), validation_y))

# 1. Get missing columns. For each missing column, add a new column "#COLUMN_NAME#_missing_value".
pre_impute_train_data = data.drop(columns=['Id','SalePrice'])
pre_impute_test_data = testing_data.drop(columns=['Id'])
testing_col_missing_value = [col for col in testing_data.columns 
                            if testing_data[col].isnull().any()]
for col in cols_with_missing_val:
    pre_impute_train_data[col+"_missing_value"] = data[col].isnull()
for col in testing_col_missing_value:
    pre_impute_test_data[col+"_missing_value"] = testing_data[col].isnull()


# 2. Normalize
pre_impute_normal_train_data = pd.get_dummies(pre_impute_train_data)
pre_impute_normal_test_data = pd.get_dummies(pre_impute_test_data)

# 3. Align
aligned_pre_impute_train_data, aligned_pre_impute_test_data = pre_impute_normal_train_data.align(
    pre_impute_normal_test_data, join="left", axis=1)

# 4. Impute
from sklearn.impute import SimpleImputer
my_imputer = SimpleImputer()
# IMPORTANT: convert fit_transform outputs to pd.DataFrame.
imputed_train_data = pd.DataFrame(
    my_imputer.fit_transform(aligned_pre_impute_train_data))
imputed_test_data = pd.DataFrame(
    my_imputer.transform(aligned_pre_impute_test_data))

# 5. Predict
y = data['SalePrice']
train_x, validation_x, train_y, validation_y = train_test_split(
    imputed_train_data, y, test_size=0.3, random_state=0)
for nodes in (50, 500, 5000, 8000):
    print('----max_leaf_nodes=', nodes, '-----')
    model = RandomForestRegressor(max_leaf_nodes=nodes)
    print('---training set error---')
    predict(model, train_x, train_y, train_x, train_y,
            max_leaf_nodes=nodes)
    print('---validation set error---')
    predict(model, train_x, train_y, validation_x, validation_y, 
            max_leaf_nodes=nodes)
    print('')

print('')
print('---no max leaf nodes---')
predict(model, train_x, train_y, train_x, train_y, max_leaf_nodes=None)
predict(model, train_x, train_y, validation_x, validation_y, 
        max_leaf_nodes=None)   

#6. Get testing result
model = RandomForestRegressor()
model.fit(train_x, train_y)
predicted_prices = model.predict(imputed_test_data)
print(predicted_prices)


