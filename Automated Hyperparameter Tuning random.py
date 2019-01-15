# -*- coding: utf-8 -*-
"""
自动超参数调优:梯度下降、贝叶斯优化引导搜索最佳超参数
绘制相同迭代次数下测试数据集 ROC 值、各项超参数变化曲线
Created on Fri Jan 11 09:24:18 2019
@author: songhu
"""
# Data manipulation
import pandas as pd
import numpy as np
import csv
import random
import lightgbm as lgb
# Visualization
import matplotlib.pyplot as plt
import seaborn as sns
import itertools
import csv
import ast

from hyperopt import STATUS_OK
from timeit import default_timer as timer
from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import roc_auc_score


plt.rcParams['font.size'] = 18
# Governing choices for search
N_FOLDS = 5
MAX_EVALS = 20

features = pd.read_csv('../input/application_train.csv')
# Sample 16000 rows (10000 for training, 6000 for testing)
features = features.sample(n = 16000, random_state = 42)
# Only numeric features
features = features.select_dtypes('number')
# Extract the labels
labels = np.array(features['TARGET'].astype(np.int32)).reshape((-1, ))
features = features.drop(columns = ['TARGET', 'SK_ID_CURR'])
# Split into training and testing data
train_features, test_features, train_labels, test_labels = train_test_split(features, labels, test_size = 6000, random_state = 42)
print('Train shape: ', train_features.shape)
print('Test shape: ', test_features.shape)
train_features.head()

# 基准模型
model = lgb.LGBMClassifier()         #random_state=50
# Training set
train_set = lgb.Dataset(train_features, label = train_labels)
test_set = lgb.Dataset(test_features, label = test_labels)

# Default hyperparamters
default_params  = model.get_params()

# Using early stopping to determine number of estimators.
del default_params ['n_estimators']
del default_params ['silent']
# Perform cross validation with early stopping
cv_results = lgb.cv(default_params , train_set, num_boost_round = 10000, nfold = N_FOLDS, metrics = 'auc', 
           early_stopping_rounds = 100, verbose_eval = False, seed = 42)
# Highest score
best = cv_results['auc-mean'][-1]
# Standard deviation of best score
best_std = cv_results['auc-stdv'][-1]
print('The maximium ROC AUC in cross validation was {:.5f} with std of {:.5f}.'.format(best, best_std))
print('The ideal number of iterations was {}.'.format(len(cv_results['auc-mean'])))

# Optimal number of esimators found in cv
model.n_estimators = len(cv_results['auc-mean'])
# Train and make predicions with model
model.fit(train_features, train_labels)
preds = model.predict_proba(test_features)[:, 1]
baseline_auc = roc_auc_score(test_labels, preds)
# print('The baseline model scores {:.5f} ROC AUC on the test set.'.format(baseline_auc))
print('基分类器 测试集 ROC 值 {:.5f} ROC AUC on the test set.'.format(baseline_auc))

def objective(hyperparameters, iteration):
    """Objective function for grid and random search. Returns
       the cross validation score from a set of hyperparameters."""
    
    # Number of estimators will be found using early stopping
    if 'n_estimators' in hyperparameters.keys():
        del hyperparameters['n_estimators']
    
     # Perform n_folds cross validation
    cv_results = lgb.cv(hyperparameters, train_set, num_boost_round = 10000, nfold = N_FOLDS, 
                        early_stopping_rounds = 100, metrics = 'auc', seed = 42)
    
    # results to retun
    score = cv_results['auc-mean'][-1]
    estimators = len(cv_results['auc-mean'])
    hyperparameters['n_estimators'] = estimators 
    
    return [score, hyperparameters, iteration]
score, params, iteration = objective(default_params, 1)
print('The cross-validation ROC AUC was {:.5f}.'.format(score))

# Hyperparameter grid
param_grid = {
    'boosting_type': ['gbdt', 'goss', 'dart'],
    'num_leaves': list(range(20, 150)),
    'learning_rate': list(np.logspace(np.log10(0.005), np.log10(0.5), base = 10, num = 1000)),
    'subsample_for_bin': list(range(20000, 300000, 20000)),
    'min_child_samples': list(range(20, 500, 5)),
    'reg_alpha': list(np.linspace(0, 1)),
    'reg_lambda': list(np.linspace(0, 1)),
    'colsample_bytree': list(np.linspace(0.6, 1, 10)),
    'subsample': list(np.linspace(0.5, 1, 100)),
    'is_unbalance': [True, False]
}


# Dataframes for random and grid search
random_results = pd.DataFrame(columns = ['score', 'params', 'iteration'],index = list(range(MAX_EVALS)))

# Create file and open connection
out_file = 'random_search_trials.csv'
of_connection = open(out_file, 'w')
writer = csv.writer(of_connection)
# Write column names
headers = ['score', 'hyperparameters', 'iteration']
writer.writerow(headers)
of_connection.close()

def random_search(param_grid, out_file, max_evals = MAX_EVALS):
    """Random search for hyperparameter optimization. 
       Writes result of search to csv file every search iteration."""  
    # Dataframe for results
    results = pd.DataFrame(columns = ['score', 'params', 'iteration'],
                                  index = list(range(MAX_EVALS)))
    for i in range(MAX_EVALS):
        
        # Choose random hyperparameters
        random_params = {k: random.sample(v, 1)[0] for k, v in param_grid.items()}
        random_params['subsample'] = 1.0 if random_params['boosting_type'] == 'goss' else random_params['subsample']

        # Evaluate randomly selected hyperparameters
        eval_results = objective(random_params, i)
        results.loc[i, :] = eval_results

        # open connection (append option) and write results
        of_connection = open(out_file, 'a')
        writer = csv.writer(of_connection)
        writer.writerow(eval_results)
        
        # make sure to close connection
        of_connection.close()
        
    # Sort with best score on top
    results.sort_values('score', ascending = False, inplace = True)
    results.reset_index(inplace = True)
    return results 

random_results = random_search(param_grid, out_file)
random_results = pd.read_csv('../input/random_search_trials.csv')
random_results['hyperparameters'] = random_results['hyperparameters'].map(ast.literal_eval)
# =============================================================================
# print('The best validation score was {:.5f}'.format(random_results.loc[0, 'score']))
# print('\nThe best hyperparameters were:')
# 
# # Get the best parameters
# random_search_params = random_results.loc[0, 'params']
# 
# # Create, train, test model
# model = lgb.LGBMClassifier(**random_search_params)
# model.fit(train_features, train_labels)
# 
# preds = model.predict_proba(test_features)[:, 1]
# print('The best model from random search scores {:.5f} ROC AUC on the test set.'.format(roc_auc_score(test_labels, preds)))
# 
# =============================================================================
def evaluate(results, name):
    """Evaluate model on test data using hyperparameters in results
       Return dataframe of hyperparameters"""        
    # Sort with best values on top
    results = results.sort_values('score', ascending = False).reset_index(drop = True)
    
    # Print out cross validation high score
    print('The highest cross validation score from {} was {:.5f} found on iteration {}.'.format(name, results.loc[0, 'score'], results.loc[0, 'iteration']))
    
    # Use best hyperparameters to create a model
    hyperparameters = results.loc[0, 'hyperparameters']
    model = lgb.LGBMClassifier(**hyperparameters)
    
    # Train and make predictions
    model.fit(train_features, train_labels)
    preds = model.predict_proba(test_features)[:, 1]
    
    print('ROC AUC from {} on test data = {:.5f}.'.format(name, roc_auc_score(test_labels, preds)))
    
    # Create dataframe of hyperparameters
    hyp_df = pd.DataFrame(columns = list(results.loc[0, 'hyperparameters'].keys()))

    # Iterate through each set of hyperparameters that were evaluated
    for i, hyp in enumerate(results['hyperparameters']):
        hyp_df = hyp_df.append(pd.DataFrame(hyp, index = [0]), 
                               ignore_index = True)
        
    # Put the iteration and score in the hyperparameter dataframe
    hyp_df['iteration'] = results['iteration']
    hyp_df['score'] = results['score']
    return hyp_df

random_hyp = evaluate(random_results, name = 'random search')

#========================== 随机搜索结果可视化 =================================
# =============================================================================
# import altair as alt
# max_random = random_hyp['score'].max()
# 
# c = alt.Chart(hyp, width = 400, height = 400).mark_circle(size = 150).encode(alt.Y('score', scale = alt.Scale(domain = [0.65, 0.76])),
# x = 'iteration', color = 'search')
# c.title = 'Score vs Iteration'
# best_random_hyp = random_hyp.iloc[random_hyp['score'].idxmax()].copy()
# 
# # Combine results into one dataframe
# random_hyp['search'] = 'random'
# grid_hyp=random_hyp.copy()
# grid_hyp['search'] = 'grid'
# hyp = random_hyp.append(grid_hyp)
# 
# #hyp.sort_values('search', inplace = True)
# # Plot of scores over the course of searching
# sns.lmplot('iteration', 'score', hue = 'search', data = hyp, size = 8);
# #plt.scatter(best_grid_hyp['iteration'], best_grid_hyp['score'], marker = '*', s = 400, c = 'blue', edgecolor = 'k')
# plt.scatter(best_random_hyp['iteration'], best_random_hyp['score'], marker = '*', s = 400, c = 'orange', edgecolor = 'k')
# 
# plt.xlabel('Iteration')
# plt.ylabel('ROC AUC')
# plt.title("Validation ROC AUC versus Iteration");
# =============================================================================
#========================== 随机搜索结果可视化 =================================
import altair as alt
max_random = random_hyp['score'].max()
c = alt.Chart(hyp, width = 400, height = 400).mark_circle(size = 150).encode(alt.Y('score', scale = alt.Scale(domain = [0.65, 0.76])),
x = 'iteration', color = 'search')
c.title = 'Score vs Iteration'
best_random_hyp = random_hyp.iloc[random_hyp['score'].idxmax()].copy()
 
# Combine results into one dataframe
random_hyp['search'] = 'random'

#hyp.sort_values('search', inplace = True)
# Plot of scores over the course of searching
sns.lmplot('iteration', 'score', hue = 'search', data = random_hyp, size = 8);
#plt.scatter(best_grid_hyp['iteration'], best_grid_hyp['score'], marker = '*', s = 400, c = 'blue', edgecolor = 'k')
plt.scatter(best_random_hyp['iteration'], best_random_hyp['score'], marker = '*', s = 400, c = 'orange', edgecolor = 'k')
 
plt.xlabel('Iteration')
plt.ylabel('ROC AUC')
plt.title("Validation ROC AUC versus Iteration");
print('Average validation score of random search = {:.5f}.'.format(np.mean(random_hyp['score'])))



# 输入完整的数据集
# Read in full dataset
train = pd.read_csv('../input/application_train.csv',nrows=10000)
test = pd.read_csv('../input/application_train.csv',nrows=5000)
train = train.select_dtypes('number')
test = test.select_dtypes('number')
# Extract the test ids and train labels
test_ids = test['SK_ID_CURR']
train_labels = np.array(train['TARGET'].astype(np.int32)).reshape((-1, ))

train = train.drop(columns = ['SK_ID_CURR', 'TARGET'])
test = test.drop(columns = ['SK_ID_CURR', 'TARGET'])

print('Training shape: ', train.shape)
print('Testing shape: ', test.shape)

random_results = pd.read_csv('../input/random_search_trials.csv')
random_results['hyperparameters'] = random_results['hyperparameters'].map(ast.literal_eval)

train_set = lgb.Dataset(train, label = train_labels)

#=========================== 贝叶斯优化搜索超参数结果===========================
hyperparameters = dict(**random_results.loc[0, 'hyperparameters'])
del hyperparameters['n_estimators']

# Cross validation with n_folds and early stopping
cv_results = lgb.cv(hyperparameters, train_set,
                    num_boost_round = 10000, early_stopping_rounds = 100, 
                    metrics = 'auc', nfold = N_FOLDS)

print('The cross validation score on the full dataset for Random optimization = {:.5f} with std: {:.5f}.'.format(
    cv_results['auc-mean'][-1], cv_results['auc-stdv'][-1]))
print('Number of estimators = {}.'.format(len(cv_results['auc-mean'])))

model = lgb.LGBMClassifier(n_estimators = len(cv_results['auc-mean']), **hyperparameters)
model.fit(train, train_labels)

preds = model.predict_proba(test)[:, 1]

#submission = pd.DataFrame({'SK_ID_CURR': test_ids, 'TARGET': preds})
#submission.to_csv('submission_bayesian_optimization.csv', index = False)
#==============================================================================





