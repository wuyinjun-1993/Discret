from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import RandomizedSearchCV
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import MultiLabelBinarizer

def gb_model_train(feat_mats, label_mats, K_CV=5, N_ITER=100, feat_count=6, multi_label=False):
    gb = GradientBoostingClassifier(verbose=1, subsample=0.9, random_state=42, n_iter_no_change=5)
    if multi_label:
        gb = MultiOutputClassifier(gb)
    print('Parameters currently in use:\n')


    max_features = ['auto', 'sqrt']
    learning_rate =  np.linspace(0.01, 0.2, num = 10)
    max_depth = [int(x) for x in np.linspace(5, feat_count, num = feat_count-5+1)]
    max_depth.append(None)
    min_samples_leaf = [1, 2, 4]
    min_samples_split = [2, 5, 10]
    n_estimators = [int(x) for x in np.linspace(start = 10, stop = 1000, num = 100)]
    subsample = [0.5, 0.8, 1.0]
    loss = ['deviance', 'exponential']

    if multi_label:
        random_grid = {'estimator__max_features': max_features,
                        'estimator__max_depth': max_depth,
                        'estimator__min_samples_leaf': min_samples_leaf,
                        'estimator__min_samples_split': min_samples_split,
                        'estimator__n_estimators': n_estimators,
                        'estimator__subsample': subsample,
                        'estimator__learning_rate': learning_rate,
                        'estimator__loss': loss}
    else:
        random_grid = {'max_features': max_features,
                        'max_depth': max_depth,
                        'min_samples_leaf': min_samples_leaf,
                        'min_samples_split': min_samples_split,
                        'n_estimators': n_estimators,
                        'subsample': subsample,
                        'learning_rate': learning_rate,
                        'loss': loss}

    gb_random = RandomizedSearchCV(estimator = gb, param_distributions = random_grid, scoring='roc_auc',
                                    n_iter = N_ITER, cv = K_CV, verbose=2, random_state=42, n_jobs = -1)
    # Fit the random search model
    # feat_mats = train_dataset.data.iloc[train_dataset.patient_ids,in_vars]
    # label_mats = train_dataset.data.iloc[train_dataset.patient_ids,"label"]
    gb_random.fit(np.array(feat_mats), np.array(label_mats))
    return gb_random

def gb_model_pred(feat_mats, gb_random, multi_label=False):
    pred_labels = gb_random.predict(feat_mats)
    pred_prob_labels = gb_random.predict_proba(feat_mats)
    if multi_label:
        pred_prob_labels = pred_prob_labels[0]
    else:
        pred_prob_labels = pred_prob_labels[:,1]
    #     pred_prob_labels = np.stack(pred_prob_labels, axis=2)
    return pred_labels, pred_prob_labels
