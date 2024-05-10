import numpy as np
import logging
from sklearn.metrics import recall_score, f1_score, roc_auc_score
import os, sys
sys.path.append(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
from utils_mortality.metrics import metrics_maps



def evaluate_performance(pred_prob_labels, test_labels, multi_label=False):
    # roc_auc_score(y_true_array, y_pred_array)
    if not multi_label:
        auc_score2 = roc_auc_score(test_labels, pred_prob_labels)
        pred_prob_labels_int = (pred_prob_labels>0.5).astype(int)
    else:
        auc_score2 = roc_auc_score(test_labels, pred_prob_labels, average=None, multi_class="ovr")
        pred_prob_labels_int = np.argmax(pred_prob_labels, axis=-1).astype(int)

    # auc_score = roc_auc_score(test_labels, (pred_prob_labels[:,1]>0.5).astype(float))

    

    accuracy = np.mean(test_labels.reshape(-1) == pred_prob_labels_int.reshape(-1))

    logger = logging.getLogger()

    # desc = "auc score::" + str(auc_score)
    # print(desc)

    # logger.log(level=logging.DEBUG, msg = desc)


    desc = "auc score ::" + str(auc_score2)
    print(desc)
    logger.log(level=logging.DEBUG, msg = desc)

    if multi_label:
        auc_score2 = np.mean(auc_score2)
        desc = "auc score mean::" + str(auc_score2)
        print(desc)
        logger.log(level=logging.DEBUG, msg = desc)

    desc = "accuracy::" + str(accuracy)
    print(desc)
    logger.log(level=logging.DEBUG, msg = desc)
    
    if not multi_label:
        additional_score_str = ""
        full_y_pred_prob_array = np.stack([1 - pred_prob_labels.reshape(-1), pred_prob_labels.reshape(-1)], axis=1)
        for metric_name in metrics_maps:
            if len(full_y_pred_prob_array.shape) == 3:
                curr_score = metrics_maps[metric_name](test_labels.reshape(-1),np.transpose(full_y_pred_prob_array, (0,2,1)).reshape(-1,2))
            else:
                curr_score = metrics_maps[metric_name](test_labels.reshape(-1),full_y_pred_prob_array)
            additional_score_str += metric_name + ": " + str(curr_score) + " "
        print(additional_score_str)
        
    return accuracy, auc_score2
