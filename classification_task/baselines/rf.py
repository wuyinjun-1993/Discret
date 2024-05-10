from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import RandomizedSearchCV
import numpy as np
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import MultiLabelBinarizer

def rf_model_train(feat_mats, label_mats, K_CV=5, N_ITER=100, feat_count=6, multi_label=False):
    print('Fiting a RandomForestClassifier')
    rf = RandomForestClassifier(oob_score=True, bootstrap=True, random_state=42)
    if multi_label:
        rf = MultiOutputClassifier(rf)
    # Look at parameters used by our current forest
    print('Starting parameters currently in use:\n')
    
    # Number of trees in random forest
    # n_estimators = [int(x) for x in np.linspace(start = 10, stop = 1000, num = 2)]
    # # Number of features to consider at every split
    # max_features = ['auto', 'sqrt']
    # # Maximum number of levels in tree
    # max_depth = [int(x) for x in np.linspace(1, feat_count, num = feat_count-1)]
    # max_depth.append(None)
    # # Minimum number of samples required to split a node
    # min_samples_split = [2, 5, 10]
    # # Minimum number of samples required at each leaf node
    # min_samples_leaf = [1, 2, 4]

    # rf = RandomForestClassifier(oob_score=True, bootstrap=True, random_state=42)

    # Look at parameters used by our current forest
    # print('Starting parameters currently in use:\n')
    # pprint(rf.get_params())

    # Number of trees in random forest
    # n_estimators = [int(x) for x in np.linspace(start = 10, stop = 1000, num = 500)]
    n_estimators = [int(x) for x in np.linspace(start = 1, stop = 40, num = 20)]
    # n_estimators = [7]
    # Number of features to consider at every split
    max_features = ['auto', 'sqrt']
    # Maximum number of levels in tree
    if int((feat_count-5)/5) <= 0:
        max_depth = [feat_count]
    else:
        max_depth = [int(x) for x in np.linspace(5, feat_count, num = int((feat_count-5)/5))]
    # max_depth=[6]
    # max_depth.append(None)
    # Minimum number of samples required to split a node
    min_samples_split = [2, 5, 10, 50, 100, 200, 500]
    # Minimum number of samples required at each leaf node
    min_samples_leaf = [1, 2, 4, 10, 50, 100, 200, 500]
    # Method of selecting samples for training each tree
    if multi_label:
        random_grid = {'estimator__n_estimators': n_estimators,
                    'estimator__max_features': max_features,
                    'estimator__max_depth': max_depth,
                    'estimator__min_samples_split': min_samples_split,
                    'estimator__min_samples_leaf': min_samples_leaf}
    else:
        random_grid = {'n_estimators': n_estimators,
                        'max_features': max_features,
                        'max_depth': max_depth,
                        'min_samples_split': min_samples_split,
                        'min_samples_leaf': min_samples_leaf}
    print(random_grid)

    # Use the random grid to search for best hyperparameters
    # Random search of parameters, using k fold cross validation,
    # search across n_iter different combinations, and use all available cores
    rf_random = RandomizedSearchCV(estimator=rf, param_distributions=random_grid, scoring='roc_auc',
                                    n_iter=N_ITER, cv=K_CV, verbose=2, random_state=42, n_jobs = -1)
    # Fit the random search model
    rf_random.fit(feat_mats, label_mats)

    return rf_random
    # Save Model
    # with open('models/rf_random_search.p', 'wb') as f:
    #     pickle.dump(rf_random, f, pickle.HIGHEST_PROTOCOL)
    # with open('models/rf_args.p', 'wb') as f:
    #     pickle.dump(args, f, pickle.HIGHEST_PROTOCOL)

def rf_model_pred(feat_mats, gb_random, multi_label=False):
    pred_labels = gb_random.predict(feat_mats)
    pred_prob_labels = gb_random.predict_proba(feat_mats)
    if multi_label:
        pred_prob_labels = pred_prob_labels[0]
    else:
        pred_prob_labels = pred_prob_labels[:,1]
    #     pred_prob_labels = np.stack(pred_prob_labels, axis=2)
    return pred_labels, pred_prob_labels