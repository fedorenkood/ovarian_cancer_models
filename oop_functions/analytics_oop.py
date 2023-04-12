from __future__ import annotations

from typing import Tuple, Callable

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn import tree
from sklearn.base import clone
from sklearn.metrics import classification_report, roc_auc_score, accuracy_score, confusion_matrix, roc_curve, auc, \
    precision_recall_curve, PrecisionRecallDisplay
import xgboost as xgb

from .classifier_oop import ClassifierDataUtil
from .merge_dataset_functions import merge_df_into_features
from .util_functions import remove_featues_startswith


def get_roc_threshold_point(fpr: np.array, tpr: np.array, thresholds: np.array) -> Tuple[float, float, float]:
    # https://stackoverflow.com/questions/28719067/roc-curve-and-cut-off-point-python
    optimal_idx = np.argmax(tpr - fpr)
    optimal_threshold = thresholds[optimal_idx]
    return float(optimal_threshold), fpr[optimal_idx], tpr[optimal_idx]


# TODO: finigh typing this one
class VisualizationUtil:
    def __init__(self) -> None:
        pass

    def display_confusion_matrix(self, ax: plt.axis, cm: pd.DataFrame) -> VisualizationUtil:
        sns.heatmap(cm, annot = True, fmt = 'd', cbar = False, ax=ax)
        ax.set_title('Confusion Matrix')
        return self

    def display_roc_graph(self, ax: plt.axis, fpr: np.array, tpr: np.array, thresholds: np.array, tpr_std: np.array = None) -> VisualizationUtil:
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr,tpr, color = '#b50000', label = 'AUC = %0.3f' % roc_auc)
        ax.plot([0, 1], [0, 1], linestyle = '-.', color = 'gray')
        if tpr_std is not None:
            tpr_upper    = np.clip(tpr+tpr_std, 0, 1)
            tpr_lower    = np.clip(tpr-tpr_std, 0, 1)
            ax.fill_between(fpr, tpr_lower, tpr_upper, color='b', alpha=.1, label='Confidence Interval')
        ax.set_ylabel('TP Rate')
        ax.set_xlabel('FP Rate')
        ax.set_title('ROC AUC Curve')
        ax.legend()
        return self

    def display_roc_threshold(self, ax: plt.axis, fpr: np.array, tpr: np.array, thresholds: np.array, tpr_std: np.array = None) -> VisualizationUtil:
        optimal_threshold, fpr_optimal, tpr_optimal = get_roc_threshold_point(fpr, tpr, thresholds)
        ax.plot(fpr_optimal, tpr_optimal,'ro', label=f'Optimal threshold: {round(optimal_threshold, 2)}')
        ax.legend()
        return self

    def display_precision_recall(self, ax: plt.axis, precision: np.array, recall: np.array, thresholds: np.array) -> VisualizationUtil:
        disp = PrecisionRecallDisplay(precision=precision, recall=recall)
        disp.plot(ax=ax)
        ax.set_title('Precision-Recall Curve')
        return self
    
class GenerateReportUtil:
    def __init__(self, y_test: np.array, y_pred: np.array, y_prob: np.array) -> None:
        self.y_test = y_test
        self.y_pred = y_pred
        self.y_pred_threshold = y_pred
        self.y_prob = y_prob
        self.report = None
        self.auc = None
        self.accuracy = None
        self.generate_report()
    
    def get_roc_threshold(self) -> float:
        fpr, tpr, thresholds = roc_curve(self.y_test, self.y_prob)
        optimal_threshold, _, _ = get_roc_threshold_point(fpr, tpr, thresholds)
        return optimal_threshold

    def apply_default_pred(self) -> GenerateReportUtil:
        self.y_pred_threshold = self.y_pred
        return self.generate_report()
    
    def apply_threshold(self, threshold: float) -> GenerateReportUtil:
        self.y_pred_threshold = np.array(pd.Series(self.y_prob).map(lambda x: 1 if x > threshold else 0))
        return self.generate_report()
    
    def apply_roc_threshold(self) -> GenerateReportUtil:
        return self.apply_roc_threshold(self.get_roc_threshold())
    
    def generate_report(self) -> GenerateReportUtil:
        report = pd.DataFrame(classification_report(self.y_test, self.y_pred_threshold, output_dict=True)).transpose().iloc[0:2 ,:]
        self.auc = roc_auc_score(self.y_test, self.y_prob)
        self.accuracy = accuracy_score(self.y_test, self.y_pred_threshold)
        report = report.drop(['support'], axis=1)
        report['class'] = report.index
        report['class'] = report['class'].astype('float')
        report['class'] = report['class'].astype('int')
        report['dummy'] = 0
        report = merge_df_into_features(report, on_col='dummy', make_unique_over_cols=['class'])
        report = remove_featues_startswith(report, ['class', 'dummy'], show_removed=False)
        report['accuracy'] = self.accuracy
        report['auc'] = self.auc
        self.report = report
        return self

    def get_report(self) -> pd.DataFrame:
        self.generate_report()
        return self.report
    
    def print_report(self) -> GenerateReportUtil:
        self.generate_report()
        print(self.report)
        print(f'ROC AUC score: {self.auc}')
        print(f'Accuracy Score: {self.accuracy}')
        return self
    
    def get_confusion_matrix(self) -> pd.DataFrame:
        return pd.DataFrame(confusion_matrix(self.y_test, self.y_pred_threshold), columns=['Predicted Healthy', 'Predicted Cancer'], index=['Healthy', 'Cancer'])
    
    def get_roc_results(self) -> Tuple[np.array, np.array, np.array]:
        fpr, tpr, thresholds = roc_curve(self.y_test, self.y_prob)
        return fpr, tpr, thresholds
    
    def get_roc_results_interp(self) -> Tuple[np.array, np.array, np.array]:
        fpr, tpr, thresholds = roc_curve(self.y_test, self.y_prob)
        fpr_mean    = np.linspace(0, 1, 100)
        interp_tpr    = np.interp(fpr_mean, fpr, tpr)
        interp_tpr[0] = 0.0
        # TODO: not sure if interpolated thresholds are even correct
        interp_thresholds = np.interp(fpr_mean, fpr, thresholds)
        interp_thresholds[0] = 0.0
        return fpr_mean, interp_tpr, interp_thresholds
    
    def get_precison_recall_results(self) -> Tuple[np.array, np.array, np.array]:
        precision, recall, thresholds = precision_recall_curve(self.y_test, self.y_prob)
        return precision, recall, thresholds
    
    def get_visualization_util(self) -> VisualizationUtil:
        return VisualizationUtil()
    
    def display_graph_interp(self) -> VisualizationUtil:
        f, ax = plt.subplots(1, 3, figsize=(16, 5))
        plt.yticks(rotation = 0)
        visualization_util = self.get_visualization_util()
        visualization_util.display_confusion_matrix(ax[0], self.get_confusion_matrix())
        visualization_util.display_roc_graph(ax[-2], *self.get_roc_results_interp())
        visualization_util.display_roc_threshold(ax[-2], *self.get_roc_results_interp())
        visualization_util.display_precision_recall(ax[-1], *self.get_precison_recall_results())
        plt.show()
        return visualization_util
    
    def display_graph(self) -> VisualizationUtil:
        f, ax = plt.subplots(1, 3, figsize=(16, 5))
        plt.yticks(rotation = 0)
        visualization_util = self.get_visualization_util()
        visualization_util.display_confusion_matrix(ax[0], self.get_confusion_matrix())
        visualization_util.display_roc_graph(ax[-2], *self.get_roc_results())
        visualization_util.display_roc_threshold(ax[-2], *self.get_roc_results())
        visualization_util.display_precision_recall(ax[-1], *self.get_precison_recall_results())
        plt.show()
        return visualization_util


class AnalyticsUtil:
    def __init__(self, classifier, data_util: ClassifierDataUtil) -> None:
        self.classifier = clone(classifier)
        self.data_util = data_util

    def store_classifier(self) -> None:
        # TODO: save trained classifier
        pass

    def load_classifier(self, filename: str) -> None:
        # TODO: load trained classifier from file
        pass

    def fit(self) -> AnalyticsUtil:
        self.data_util.check_if_data_util_initialized()

        X_train, y_train = self.data_util.get_train_data()
        self.classifier.fit(X_train, y_train)
        return self

    def get_predictions_general(self, X_test: pd.DataFrame) -> Tuple[np.array, np.array]:
        y_pred = self.classifier.predict(X_test)
        y_prob = self.classifier.predict_proba(X_test)[: ,1]
        return y_pred, y_prob

    def get_predictions(self) -> Tuple[np.array, np.array]:
        X_test, y_test = self.data_util.get_test_data()
        return self.get_predictions_general(X_test)
    
    def get_report_generation_util(self) -> GenerateReportUtil:
        X_test, y_test = self.data_util.get_test_data()
        y_pred, y_prob = self.get_predictions() 
        return GenerateReportUtil(y_test, y_pred, y_prob)
    
    def get_report_generation_util_filtered(self, filter: Callable[[pd.DataFrame], pd.DataFrame]) -> GenerateReportUtil:
        X_test, y_test = self.data_util.get_filtered_test_data(filter)
        y_pred, y_prob = self.get_predictions_general(X_test) 
        return GenerateReportUtil(y_test, y_pred, y_prob)

    def plot_save_tree(self, plot_tree=False, filepath=None) -> AnalyticsUtil:
        return self

    def feature_selection(self):
        pass

class TreeAnalyticsUtil(AnalyticsUtil):
    def get_max_depth(self) -> int:
        return 0
    
    def feature_selection(self):
    
        feature_importances = pd.DataFrame(self.classifier.feature_importances_,
                            index = self.data_util.get_feature_names(),
                            columns=['importance']).sort_values('importance', 
                                                                ascending=False)
        feature_importances['column_name'] = feature_importances.index
        feature_importances = feature_importances[['column_name', 'importance']]
        # TODO: boundary and confusion matrix
        tree_depth = self.get_max_depth()
        # TODO: results for filtered tests?
        report_util = self.get_report_generation_util()
        top_feature_stats = {
            'top_feature': feature_importances.iloc[0]['column_name'],
            # 'boundary': 0,
            'num_features_used': len(feature_importances[feature_importances['importance'] > 0]),
            'importance': round(feature_importances.iloc[0]['importance'], 3),
            'tree_depth': tree_depth,
            'accuracy': report_util.accuracy,
            'auc': report_util.auc,
            # 'precision': precision,
            # 'recall':    recall,
            # 'f1-score':  f1,
        }
        return top_feature_stats, feature_importances
    
class DesicionTreeAnalyticsUtil(TreeAnalyticsUtil):
    def plot_save_tree(self, plot_tree=False, filepath=None) -> DesicionTreeAnalyticsUtil:
        if plot_tree or filepath is not None:
            cn=['no cancer', 'cancer']
            fig, axes = plt.subplots(nrows = 1,ncols = 1, dpi=3000)
            tree.plot_tree(self.classifier,
                        max_depth=5,
                feature_names = self.data_util.get_feature_names(), 
                class_names=cn,
                filled = True)
            if filepath is not None:
                plt.savefig(filepath)
            if plot_tree:
                plt.show()
        return self

    def get_max_depth(self) -> int:
        return self.classifier.tree_.max_depth
    
class XgbAnalyticsUtil(TreeAnalyticsUtil):
    def plot_save_tree(self, plot_tree=False, filepath=None) -> XgbAnalyticsUtil:
        if plot_tree or filepath is not None:
            cn=['no cancer', 'cancer']
            fig, axes = plt.subplots(nrows = 1,ncols = 1, dpi=6000)
            # plt.gcf().set_size_inches(18.5, 10.5)
            xgb.plot_tree(self.classifier, rankdir='LR', ax=axes)
            if filepath is not None:
                plt.savefig(filepath)
            if plot_tree:
                plt.show()
        return self

    def get_max_depth(self) -> int:
        return self.classifier.max_depth
