from sklearn.metrics import brier_score_loss
import numpy as np
# from pycaleva import CalibrationEvaluator
import os,sys
sys.path.append(os.path.dirname(os.path.realpath(__file__)))
from general_calibration_score import ace
def ece_score(y_test, py, n_bins=10):
    py = np.array(py)
    y_test = np.array(y_test)
    if y_test.ndim > 1:
        y_test = np.argmax(y_test, axis=1)
    py_index = np.argmax(py, axis=1)
    py_value = py[np.arange(len(py)), py_index]
    acc, conf = np.zeros(n_bins), np.zeros(n_bins)
    Bm = np.zeros(n_bins)
    for m in range(n_bins):
        a, b = m / n_bins, (m + 1) / n_bins
        for i in range(py.shape[0]):
            if py_value[i] > a and py_value[i] <= b:
                Bm[m] += 1
                if py_index[i] == y_test[i]:
                    acc[m] += 1
                conf[m] += py_value[i]
        if Bm[m] != 0:
            acc[m] = acc[m] / Bm[m]
            conf[m] = conf[m] / Bm[m]
    ece = 0
    for m in range(n_bins):
        ece += Bm[m] * np.abs((acc[m] - conf[m]))
    return ece / sum(Bm)

def ace_scores(y_test, py):
    # pred_y_labels = np.argmax(py, axis=1)

    best_y_prob = py[:,1]#py[np.arange(len(py)), pred_y_labels]
    if ( y_test.sum() <= 1 ) or ( y_test.sum() >= (len(y_test) - 1) ):
        return -1
    res = ace(y_test.astype(int).reshape(-1), best_y_prob.reshape(-1))
    return res

    # ce = CalibrationEvaluator(y_test, best_y_prob, outsample=True, n_groups='auto')
    # return ce.ace

# def mean_absolute_error(
#     y_true, y_pred, *, sample_weight=None, multioutput="uniform_average"):
#     y_type, y_true, y_pred, multioutput = _check_reg_targets(
#         y_true, y_pred, multioutput
#     )
#     check_consistent_length(y_true, y_pred, sample_weight)
#     output_errors = np.average(np.abs(y_pred - y_true), weights=sample_weight, axis=0)
#     if isinstance(multioutput, str):
#         if multioutput == "raw_values":
#             return output_errors
#         elif multioutput == "uniform_average":
#             # pass None as weights to np.average: uniform mean
#             multioutput = None

#     return np.average(output_errors, weights=multioutput)

# def integrated_calibration_index(y, p, **args):

#     smoothed = lowess(y, p, **args)
#     return mean_absolute_error(*smoothed.transpose())


def calculate_brier_score(y_true, y_prob):
    # pred_y_labels = y_prob[:,1]

    best_y_prob = y_prob[:, 1]

    return brier_score_loss(y_true, best_y_prob.reshape(-1))
# to be checked:
def integrated_calibration_index(y_true, y_prob, num_bins=10):
    # Convert inputs to numpy arrays
    y_true = np.asarray(y_true)
    pred_y_labels = np.argmax(y_prob, axis=1)
    best_y_prob = y_prob[np.arange(len(y_prob)), pred_y_labels]
    # y_prob = np.asarray(y_prob)

    # Divide the predicted probabilities into equally sized bins
    bin_indices = np.digitize(best_y_prob, np.linspace(0, 1, num_bins + 1))

    # Calculate the fraction of true positives and true negatives in each bin
    tp_fraction = np.mean((y_true == 1) & (bin_indices == np.expand_dims(np.arange(1, num_bins + 1), axis=1)), axis=1)
    tn_fraction = np.mean((y_true == 0) & (bin_indices == np.expand_dims(np.arange(1, num_bins + 1), axis=1)), axis=1)

    # Calculate the calibration error in each bin
    calibration_error = np.abs(tp_fraction - tn_fraction)

    # Calculate the integrated calibration index
    ici = np.sum(calibration_error) / (num_bins - 1)

    return ici

def print_additional_metrics(y_pred_prob_array, y_true_array):
    additional_score_str = ""
    full_y_pred_prob_array = np.stack([1 - y_pred_prob_array.reshape(-1), y_pred_prob_array.reshape(-1)], axis=1)
    for metric_name in metrics_maps:
        curr_score = metrics_maps[metric_name](y_true_array.reshape(-1),full_y_pred_prob_array)
        additional_score_str += metric_name + ": " + str(curr_score) + " "

    print(additional_score_str)


metrics_maps = {"brier_score_loss": calculate_brier_score, "Expected Calibration Error": ece_score, "Adaptive Calibration Error": ace_scores}