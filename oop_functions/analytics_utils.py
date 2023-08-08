from oop_functions.analytics_cv_util import *


def load_cv_analytics_util_see_stats(filesuffix):
    cv_analytics_util = CvAnalyticsUtil.load_cv_analytics_utils(filesuffix)
    cv_analytics_util.get_cv_report();
    cv_analytics_util.display_graph()
    return cv_analytics_util


def evaluate_threshold(threshold, predictions, targets):
    # Create binary predictions based on the threshold
    binary_predictions = (predictions >= threshold).astype(int)
    
    # Calculate evaluation metrics
    tp = np.sum((binary_predictions == 1) & (targets == 1))
    fp = np.sum((binary_predictions == 1) & (targets == 0))
    tn = np.sum((binary_predictions == 0) & (targets == 0))
    fn = np.sum((binary_predictions == 0) & (targets == 1))
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    true_positive_rate = tp / (tp + fn) if (tp + fn) > 0 else 0
    false_positive_rate = fp / (fp + tn) if (fp + tn) > 0 else 0
    
    return tp, fp, tn, fn, precision, recall, true_positive_rate, false_positive_rate


def get_per_thereshold_metrics(df, probability, label, thresholds = np.linspace(0, 1, 101)):
    # Initialize lists to store results for each threshold
    tp_list = []
    fp_list = []
    tn_list = []
    fn_list = []
    precision_list = []
    recall_list = []
    tpr_list = []
    fpr_list = []

    # Iterate through each threshold and calculate metrics
    for threshold in thresholds:
        tp, fp, tn, fn, precision, recall, tpr, fpr = evaluate_threshold(threshold, df[f'{label}_prob'], df[label])
        
        tp_list.append(tp)
        fp_list.append(fp)
        tn_list.append(tn)
        fn_list.append(fn)
        precision_list.append(precision)
        recall_list.append(recall)
        tpr_list.append(tpr)
        fpr_list.append(fpr)

    # Create a new DataFrame to store the results
    results_df = pd.DataFrame({
        'Threshold': thresholds,
        'True_Positive': tp_list,
        'False_Positive': fp_list,
        'True_Negative': tn_list,
        'False_Negative': fn_list,
        'Precision': precision_list,
        'Recall': recall_list,
        'True_Positive_Rate': tpr_list,
        'False_Positive_Rate': fpr_list
    })
    return results_df

