import pandas as pd
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss
import matplotlib.pyplot as plt

path = '../input/house-prices-advanced-regression-techniques/train.csv'
# read the data and store data in DataFrame titled melbourne_data
data = pd.read_csv(path) 
testing_data = pd.read_csv('../input/house-prices-advanced-regression-techniques/test.csv')

def predict(model, training_x, training_y, 
            validation_x, validation_y):
    model.fit(training_x, training_y)
    print('error:', 
          mean_absolute_error(
              model.predict(validation_x), validation_y))

# Prepare data.
cols_with_missing_val = [col for col in data.columns 
                         if data[col].isnull().any()]
pre_impute_train_data = data.drop(columns=['Id','SalePrice'])
pre_impute_test_data = testing_data.drop(columns=['Id'])
testing_col_missing_value = [col for col in testing_data.columns 
                            if testing_data[col].isnull().any()]
for col in cols_with_missing_val:
    pre_impute_train_data[col+"_missing_value"] = data[col].isnull()
for col in testing_col_missing_value:
    pre_impute_test_data[col+"_missing_value"] = testing_data[col].isnull()


# Normalize
pre_impute_normal_train_data = pd.get_dummies(pre_impute_train_data)
pre_impute_normal_test_data = pd.get_dummies(pre_impute_test_data)

# Align
aligned_pre_impute_train_data, aligned_pre_impute_test_data = pre_impute_normal_train_data.align(
    pre_impute_normal_test_data, join="left", axis=1)

# Split
y = data['SalePrice']
pre_impute_train_x, pre_impute_validation_x, train_y, validation_y = train_test_split(
    aligned_pre_impute_train_data, y, test_size=0.3, random_state=0)

# Impute
from sklearn.impute import SimpleImputer
my_imputer = SimpleImputer(strategy='median')
train_x = pd.DataFrame(
    my_imputer.fit_transform(pre_impute_train_x))
validation_x = pd.DataFrame(
    my_imputer.transform(pre_impute_validation_x))
imputed_test_data = pd.DataFrame(
    my_imputer.transform(aligned_pre_impute_test_data))
    
# Predict with GradientBoostingRegressor
from sklearn.ensemble.partial_dependence import partial_dependence, plot_partial_dependence
from sklearn.ensemble import GradientBoostingRegressor

for rounds in (500, 1000, 2000):
    print('----n_estimators=', rounds, '-----')
    model = GradientBoostingRegressor(n_estimators=rounds, learning_rate=0.05)
    model.fit(train_x, train_y)
    print('---training set error---')
    print(mean_absolute_error(model.predict(train_x), train_y))
    print('---validation set error---')
    print(mean_absolute_error(model.predict(validation_x), validation_y))
    print('')
    
#=====plot=======
fig = plt.figure()
ax1 = fig.add_subplot(111)
ax1.scatter(model.predict(train_x), train_y, s=1, c='b', marker="s", label='real')
ax1.scatter(model.predict(validation_x),validation_y, s=10, c='r', marker="o")
plt.show()

#=====residual graph========
plt.scatter(predicted_train_y, predicted_train_y-train_y)

#=====plot_partial_dependence====
my_plots = plot_partial_dependence(my_model,       
                                   features=[0, 1, 2, 3], # column numbers of plots we want to show
                                   X=train_x,       # raw predictors data.
                                   feature_names=['YearBuilt', 'LotArea', 'LandSlope', 'BsmtFinSF1'], # labels on graphs
                                   #feature_names=list(train_x),
                                   grid_resolution=10) # number of values to plot on x axis
#=====predict test========
model = GradientBoostingRegressor(n_estimators=1000, learning_rate=0.05)
model.fit(train_x, train_y)
predicted_prices = model.predict(imputed_test_data)

#=====write to output=======
my_submission = pd.DataFrame({'Id': testing_data.Id, 'SalePrice': predicted_prices})
# you could use any filename. We choose submission here
my_submission.to_csv('submission_gradient_boost.csv', index=False)
