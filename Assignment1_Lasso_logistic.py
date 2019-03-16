#%%
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score

train_path = r'.\telco_train.csv'
test_path = r'.\telco_test.csv'

train_data = pd.read_csv(train_path)
test_data = pd.read_csv(test_path)
#%%
# Check missingness of dataset
train_data.isnull().sum()
test_data.isnull().sum()
# Features with high missingness cannot be used
# FIN_STATE, AVG_DATA_3MONTH, COUNT_CONNECTIONS_3MONTH, AVG_DATA_1MONTH 
#%%
# Create a column of active_days
train_data['Current_date'] = '2013-10-31' 
test_data['Current_date'] = '2013-10-31'
train_data[['START_DATE', 'Current_date']] = train_data[['START_DATE', 'Current_date']].astype('datetime64[ns]') 
test_data[['START_DATE', 'Current_date']] = test_data[['START_DATE', 'Current_date']].astype('datetime64[ns]')
 
train_data['Active_days'] = (train_data['Current_date'] - train_data['START_DATE']).dt.days
test_data['Active_days'] = (test_data['Current_date'] - test_data['START_DATE']).dt.days
#%%
# Features will be used
features_total = train_data.columns.values.tolist()
features_drop = ['ID', 'CHURN', 'START_DATE', 'Current_date', 'FIN_STATE',
    'AVG_DATA_3MONTH', 'COUNT_CONNECTIONS_3MONTH', 'AVG_DATA_1MONTH']
features_used = list(set(features_total) - set(features_drop))
#%%
# Create a preprocessing pipeline
class DataFrameSelector(BaseEstimator, TransformerMixin):
    '''
    Select and preprocess features from dataframe
    '''
    def __init__(self, feature_name):
        self.feature_name = feature_name
    
    def fit(self, df, y=None):
        return self
    
    def transform(self, df, y=None):
        return df[self.feature_name].values

# Drop binary feature 'PREPAID'    
num_feats = features_used.copy()
num_feats.remove('PREPAID')
      
# Numeric features processing pipeline    
num_feats_pipeline = Pipeline([
    ('num_feats_selector', DataFrameSelector(num_feats)),
    ('scaler', StandardScaler())
])
    
train_data_num = num_feats_pipeline.fit_transform(train_data)
test_data_num = num_feats_pipeline.transform(test_data)

# Stack standardized numeric features and binary features
X_train = np.hstack((train_data_num, train_data[['PREPAID']].values))
X_test = np.hstack((test_data_num, test_data[['PREPAID']].values))

# Create target values
y_train = train_data['CHURN'].values
#%%
# Create Lasso-logistic learning algorithm
logreg_lib = LogisticRegression(penalty='l1', solver='liblinear')
para_c = np.logspace(-1.5, 1, num=26)
grid_params = [{'C': para_c}]
accuracy_list_lib = []
best_c_list_lib = [] 

for i in range(6, 11):
    grid_search_lib = GridSearchCV(logreg_lib, grid_params, cv=i) 
    grid_search_lib.fit(X_train, y_train)
    
    # Model evaluating
    y_train_pred = grid_search_lib.predict(X_train)
    accuracy_list_lib.append(accuracy_score(y_train, y_train_pred))
    best_c_list_lib.append(grid_search_lib.best_params_['C'])
# Close accuracy_score is found under each value of i
# i = 8, 10 both have the highest accuracy of 0.8014
#%%
# Retrieve predicted probability
y_pred = grid_search_lib.predict_proba(X_test)[:, 1]