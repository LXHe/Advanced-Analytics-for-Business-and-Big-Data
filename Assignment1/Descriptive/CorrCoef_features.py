#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr

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
train_data['Current_date'] = '2013-10-31' 
test_data['Current_date'] = '2013-10-31'
train_data[['START_DATE', 'Current_date']] = train_data[['START_DATE', 'Current_date']].astype('datetime64[ns]') 
test_data[['START_DATE', 'Current_date']] = test_data[['START_DATE', 'Current_date']].astype('datetime64[ns]')
#train_data['Current_date'] = train_data['Current_date'].astype('datetime64[ns]') 
#test_data['Current_date'] = test_data['Current_date'].astype('datetime64[ns]')
 
train_data['Active_days'] = (train_data['Current_date'] - train_data['START_DATE']).dt.days
test_data['Active_days'] = (test_data['Current_date'] - test_data['START_DATE']).dt.days

#%%
features_total = train_data.columns.values.tolist()
features_drop = ['ID', 'CHURN', 'START_DATE', 'Current_date', 'FIN_STATE',
    'AVG_DATA_3MONTH', 'COUNT_CONNECTIONS_3MONTH', 'AVG_DATA_1MONTH']
features_used = list(set(features_total) - set(features_drop))
num_feats = features_used.copy()
num_feats.remove('PREPAID')

#%%
# Correlation Plot
# Set transparency so that overlapping points will become darker
scatter_matrix(train_data[num_feats], alpha=0.3)

#%%
# Check correlations between numeric features
feats_corr_matrix = train_data[num_feats].corr().round(3)

def calculate_pvalues(train_data, num_feats):
    '''
    Create a pvalue matrix
    '''
    df_feats = pd.DataFrame(columns=num_feats)
    pvalues_feats = df_feats.transpose().join(df_feats, how='outer')

    # pearsonr [0] gives correlation coefficient, [1] gives pvalue
    for i in num_feats:
        for j in num_feats:
            pvalues_feats[i][j] = round(pearsonr(train_data[i], train_data[j])[1], 3)
    return pvalues_feats

pvalues_matrix = calculate_pvalues(train_data, num_feats)

# Create masks
mask1 = feats_corr_matrix.applymap(lambda x: '{}*'.format(x))
mask2 = feats_corr_matrix.applymap(lambda x: '{}**'.format(x))
mask3 = feats_corr_matrix.applymap(lambda x: '{}***'.format(x))

marked_feats_corr = feats_corr_matrix.mask(pvalues_matrix <= 0.05, mask1)
marked_feats_corr = feats_corr_matrix.mask(pvalues_matrix <= 0.01, mask2)
marked_feats_corr = feats_corr_matrix.mask(pvalues_matrix <= 0.001, mask3)

#%%
# Plots of feature distribution
for i, feat in enumerate(num_feats):
    axes = plt.subplot(6,6,i+1)
    sns.distplot(train_data[[feat]], ax=axes)
    
plt.suptitle('Distribution of numeric features')
plt.show()
