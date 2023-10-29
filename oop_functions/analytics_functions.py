import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.metrics import classification_report, roc_auc_score, accuracy_score, confusion_matrix, roc_curve, auc, \
    precision_recall_curve, PrecisionRecallDisplay
import seaborn as sns

from .merge_dataset_functions import merge_df_into_features
from .util_functions import remove_featues_startswith


def run_classifier(classifier, X_train, X_test, y_train, y_test, show_graph=True):
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)
    y_prob = classifier.predict_proba(X_test)[: ,1]

    auc, accuracy, threshold, report = performance_analysis(y_pred, y_prob, y_test, show_graph=show_graph)

    print(f'Threshold: {threshold}')
    # Find prediction to the dataframe applying threshold
    y_pred_threathold = np.array(pd.Series(y_prob).map(lambda x: 1 if x > threshold else 0))
    performance_analysis(y_pred_threathold, y_prob, y_test, show_graph=show_graph)
    return classifier, auc, accuracy, threshold, report


def performance_analysis(y_pred, y_prob, y_test, show_graph=True):
    report = pd.DataFrame(classification_report(y_test, y_pred, output_dict=True)).transpose().iloc[0:2 ,:]
    auc = roc_auc_score(y_test, y_prob)
    accuracy = accuracy_score(y_test, y_pred)
    fpr, tpr, thresholds = roc_curve(y_test, y_pred)
    optimal_idx = np.argmax(tpr - fpr)
    threshold = thresholds[optimal_idx]
    if show_graph:
        print(report)
        print(f'ROC AUC score: {auc}')
        print(f'Accuracy Score: {accuracy}')
        f, ax = plt.subplots(1, 3, figsize=(16, 5))
        plt.yticks(rotation = 0)
        display_confusion_matrix(ax[0], y_test, y_pred)

        threshold = display_auc(ax[-2], y_test, y_prob)
        display_precision_recall(ax[-1], y_prob, y_test)
        plt.show()
    report = report.drop(['support'], axis=1)
    report['class'] = report.index
    report['class'] = report['class'].astype('float')
    report['class'] = report['class'].astype('int')
    report['dummy'] = 0
    report = merge_df_into_features(report, on_col='dummy', make_unique_over_cols=['class'])
    report = remove_featues_startswith(report, ['class', 'dummy'], show_removed=False)
    report['accuracy'] = accuracy
    report['auc'] = auc
    return auc, accuracy, threshold, report


def get_accuracy(y_test, y_pred):
    return accuracy_score(y_test, y_pred ) *100


def display_confusion_matrix(ax, y_test, y_pred):
    cm = pd.DataFrame(confusion_matrix(y_test, y_pred), columns=['Predicted Healthy', 'Predicted Cancer'], index=['Healthy', 'Cancer'])
    sns.heatmap(cm, annot = True, fmt = 'd', cbar = False, ax=ax)
    # ax.set_title('Confusion Matrix')
    # ax.set_yticks(rotation = 0)


def display_auc(ax, y_test, y_pred):
    fpr, tpr, thresholds = roc_curve(y_test, y_pred)
    i = np.arange(len(tpr))
    roc = pd.DataFrame({'tf' : pd.Series(tpr-(1-fpr), index=i), 'threshold' : pd.Series(thresholds, index=i)})
    roc_t = roc.iloc[(roc.tf-0).abs().argsort()[:1]]
    optimal_idx = np.argmax(tpr - fpr)
    optimal_threshold = thresholds[optimal_idx]
    print(optimal_threshold)

    roc_auc = auc(fpr, tpr)
    ax.plot(fpr,tpr, color = '#b50000', label = 'AUC = %0.3f' % roc_auc)
    ax.plot([0, 1], [0, 1], linestyle = '-.', color = 'gray')
    ax.plot(fpr[optimal_idx],tpr[optimal_idx],'ro', label='Optimal point')
    ax.set_ylabel('TP Rate')
    ax.set_xlabel('FP Rate')
    # ax.set_title('ROC AUC Curve')
    ax.legend()
    # return list(roc_t['threshold'])
    return optimal_threshold


def display_precision_recall(ax, y_pred, y_test):
    precision, recall, _ = precision_recall_curve(y_test, y_pred)
    disp = PrecisionRecallDisplay(precision=precision, recall=recall)
    disp.plot(ax=ax)
    # ax.set_title('Precision-Recall Curve')