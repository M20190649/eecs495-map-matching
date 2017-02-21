# Imports
        
import os
import xgboost as xgb
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


from datetime import datetime


# Constants
DATA_DIR = "probe_data_map_matching"

link_headers = ["linkPVID", 
                "refNodeID", 
                "nrefNodeID", 
                "length", 
                "functionalClass", 
                "directionOfTravel", 
                "speedCategory", 
                "fromRefSpeedLimit", 
                "toRefSpeedLimit", 
                "fromRefNumLanes", 
                "toRefNumLanes", 
                "multiDigitized", 
                "urban", 
                "timeZone", 
                "shapeInfo", 
                "curvatureInfo", 
                "slopeInfo"]

# Read in the Link Data
link_data = pd.read_csv(os.path.join(DATA_DIR, "Partition6467LinkData.csv"), header=None, names=link_headers)
link_data.drop('curvatureInfo', axis=1, inplace=True)

print(link_data.shape)

link_data.head()

# how many have slope info?
link_data.dropna().shape

# Actually get rid of the missings
link_data.dropna(inplace=True)

# Read in the (Matched) Probe Data

probe_data = pd.read_csv(os.path.join(DATA_DIR, "heading_match.csv"))
probe_data['dateTime'] = pd.to_datetime(probe_data['dateTime'], format='%Y-%m-%d %H:%M:%S')

# How many unique linkPVID's are contained in the matched probe data
print("Roughly what do the Groups look like?")
itr = 0
for i in probe_data.groupby('linkPVID').groups:
    if itr <= 5:
        print(i, probe_data.groupby('linkPVID').groups[i])
        itr += 1

print("There are {} Unique linkPVID's represented".format(len(probe_data.groupby('linkPVID').groups)))

joined_data = probe_data.join(link_data.set_index('linkPVID'), on='linkPVID', how='inner', lsuffix='l', rsuffix='r')


print("And now after the join and NA removal, we have {} rows of data".format(joined_data.shape[0]))

def rolling_diff(df):
    return df[-1] - df[0]

cleaned_joined_data = joined_data.dropna()

cleaned_joined_data['avg_link_slope'] = cleaned_joined_data['slopeInfo'].apply(lambda x: np.mean([float(i.split('/')[1]) for i in x.split('|')]))

gdf = cleaned_joined_data.groupby('sampleID')
cleaned_joined_data['delta_elevation'] = gdf['altitude'].apply(lambda x: x.rolling(2, min_periods=2).apply(rolling_diff)).reset_index(0, drop=True)
cleaned_joined_data['delta_latitude'] = gdf['latitude'].apply(lambda x: x.rolling(2, min_periods=2).apply(rolling_diff)).reset_index(0, drop=True)
cleaned_joined_data['delta_longitude'] = gdf['longitude'].apply(lambda x: x.rolling(2, min_periods=2).apply(rolling_diff)).reset_index(0, drop=True)
cleaned_joined_data['delta_speed'] = gdf['speed'].apply(lambda x: x.rolling(2, min_periods=2).apply(rolling_diff)).reset_index(0, drop=True)
cleaned_joined_data['rolling_slope'] = cleaned_joined_data['delta_elevation'] / np.sqrt(cleaned_joined_data['delta_latitude'] ** 2 + cleaned_joined_data['delta_longitude'] ** 2)
cleaned_joined_data['rolling_acc'] = cleaned_joined_data['delta_speed'] / np.sqrt(cleaned_joined_data['delta_latitude'] ** 2 + cleaned_joined_data['delta_longitude'] ** 2)

cleaned_joined_data = cleaned_joined_data.dropna()



cleaned_joined_data['speed_limit_diff'] = cleaned_joined_data['speed'] - cleaned_joined_data['fromRefSpeedLimit']

cleaned_joined_data['multiDigitized'] = cleaned_joined_data['multiDigitized'].apply(lambda x: 1 if x in 'T' else 0)



feature_set = [#'sampleID',
                   #'dateTime',
                   'altitude',
                    'latitude', 
                    'longitude',
                   'speed', 
                   'heading', 
                   'length',
                   #'functionalClass',
                   #'fromRefSpeedLimit',
                    'speed_limit_diff',
                   #'toRefSpeedLimit', 
                   #'fromRefNumLanes', 
                   #'toRefNumLanes',
                    #'multiDigitized',
                    #'delta_elevation',
                    'distFromLink',
                    'delta_speed',
                    'rolling_acc',
                    'rolling_slope',
                   'avg_link_slope']

subset = cleaned_joined_data[feature_set]

print(subset.shape)
import sklearn
train, test = sklearn.cross_validation.train_test_split(subset, test_size = 0.2, random_state=1)


dtrain = xgb.DMatrix(train.values[:,:-1], train.values[:,-1], feature_names = feature_set[:-1])
dtest = xgb.DMatrix(test.values[:,:-1], test.values[:,-1], feature_names = feature_set[:-1])
param = {'max_depth':10, 'eta':0.2, 'silent':1, "lambda": 1.2, "objective": "reg:linear", "booster":"gbtree" }
param['nthread'] = 63
param['eval_metric'] = 'rmse'
evallist  = [(dtest,'eval'), (dtrain,'train')]
num_round = 2200
bst = xgb.train( param, dtrain, num_round, evallist, early_stopping_rounds=10)

xgb.plot_importance(bst)

preds = bst.predict(dtest)
zipped_preds = test
zipped_preds['preds'] = preds

plt.figure()
plt.scatter(zipped_preds['avg_link_slope'], zipped_preds['preds'])
plt.xlabel('avg link slope')
plt.ylabel('predicted link slope')
plt.show()
