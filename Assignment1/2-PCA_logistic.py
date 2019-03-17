#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA

train_path = r'.\telco_train.csv'
test_path = r'.\telco_test.csv'

train_data = pd.read_csv(train_path)
test_data = pd.read_csv(test_path)

#%%
train_data['Current_date'] = '2013-10-31' 
test_data['Current_date'] = '2013-10-31'
train_data[['START_DATE', 'Current_date']] = train_data[['START_DATE', 'Current_date']].astype('datetime64[ns]') 
test_data[['START_DATE', 'Current_date']] = test_data[['START_DATE', 'Current_date']].astype('datetime64[ns]')
 
train_data['Active_days'] = (train_data['Current_date'] - train_data['START_DATE']).dt.days
test_data['Active_days'] = (test_data['Current_date'] - test_data['START_DATE']).dt.days

#%%
features_total = train_data.columns.values.tolist()
features_drop = ['ID', 'CHURN', 'START_DATE', 'Current_date', 'FIN_STATE',
    'AVG_DATA_3MONTH', 'COUNT_CONNECTIONS_3MONTH', 'AVG_DATA_1MONTH']
features_used = list(set(features_total) - set(features_drop))

# Drop binary feature 'PREPAID'    
num_feats = features_used.copy()
num_feats.remove('PREPAID')

#%%
# Check variance explained by PCA features
pca = PCA()
pca.fit(train_data[features_used])
pca_feats = range(pca.n_components_)
plt.bar(pca_feats, pca.explained_variance_)
ax = plt.gca()
plt.xticks(pca_feats)
plt.text(
    0.07, 0.97,
    'Pct of explained variance:\n{:.1%}'.format(pca.explained_variance_[0]/pca.explained_variance_.sum()),
    ha = 'center', va = 'center',
    transform = ax.transAxes
)

plt.annotate(
    'Pct of explained variance:\n{:.1%}'.format(pca.explained_variance_[1]/pca.explained_variance_.sum()),
    xy = (0.09, 0.01),
    xytext = (0.07, 0.07),
    xycoords = ax.transAxes,
    textcoords = ax.transAxes,
    arrowprops=dict(arrowstyle="->")      
)

plt.annotate(
    'Pct of explained variance:\n{:.1%}'.format(pca.explained_variance_[2]/pca.explained_variance_.sum()),
    xy = (0.12, 0.01),
    xytext = (0.15, 0.03),
    xycoords = ax.transAxes,
    textcoords = ax.transAxes,
    arrowprops=dict(arrowstyle="->")      
)

plt.ylabel('Explained variance')
plt.xlabel('PCA feature')
plt.title('Explained variance by PCA features')
plt.show()
# From the graph, the first feature explains a substantial amount of variance

#%%
# Binary varaible included
# Build preprocessing pipeline
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

# MinMaxScaler
num_feats_pipeline = Pipeline([
    ('num_feats_selector', DataFrameSelector(num_feats)),
    ('scaler', MinMaxScaler())
])

#%%
# Create training and test sets
train_data_num = num_feats_pipeline.fit_transform(train_data)
test_data_num = num_feats_pipeline.transform(test_data)

# Stack standardized numeric features and binary features
X_train_stack = np.hstack((train_data_num, train_data[['PREPAID']].values))
X_test_stack = np.hstack((test_data_num, test_data[['PREPAID']].values))

# PCA decomposition
pca = PCA(n_components = 3)
X_train = pca.fit_transform(X_train_stack)
X_test = pca.transform(X_test_stack)

# Creat target values
y_train = train_data['CHURN'].values

#%%
# Creat learning algorithm
# Lasso-logistic solver='liblinear'
logreg_lib = LogisticRegression(penalty='l1', solver='liblinear')
para_c = np.logspace(-1, 1, num=41)
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
# same accuracy when i is greater than 6
# i=10 and n_components=3 give the highest accuracy among all pca_logistic regression models (with different n_components)
# accuracy = 0.7752 which is smaller than the previous lass_logistic regression model
