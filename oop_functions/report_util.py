from __future__ import annotations

from typing import Tuple

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.metrics import classification_report, roc_auc_score, accuracy_score, confusion_matrix, roc_curve, \
    precision_recall_curve

from .merge_dataset_functions import merge_df_into_features
from .util_functions import remove_featues_startswith, get_roc_threshold_point
from .visualization_util import VisualizationUtil


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
        report = pd.DataFrame(
            classification_report(self.y_test, self.y_pred_threshold, output_dict=True)).transpose().iloc[0:2, :]
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
        return pd.DataFrame(confusion_matrix(self.y_test, self.y_pred_threshold),
                            columns=['Predicted Healthy', 'Predicted Cancer'], index=['Healthy', 'Cancer'])

    def get_roc_results(self) -> Tuple[np.array, np.array, np.array]:
        fpr, tpr, thresholds = roc_curve(self.y_test, self.y_prob)
        return fpr, tpr, thresholds

    def get_roc_results_interp(self) -> Tuple[np.array, np.array, np.array]:
        fpr, tpr, thresholds = roc_curve(self.y_test, self.y_prob)
        fpr_mean = np.linspace(0, 1, 100)
        interp_tpr = np.interp(fpr_mean, fpr, tpr)
        interp_tpr[0] = 0.0
        # TODO: not sure if interpolated thresholds are even correct
        interp_thresholds = np.interp(fpr_mean, fpr, thresholds)
        interp_thresholds[0] = 0.0
        return fpr_mean, interp_tpr, interp_thresholds

    def get_precison_recall_results(self) -> Tuple[np.array, np.array, np.array]:
        precision, recall, thresholds = precision_recall_curve(self.y_test, self.y_prob)
        return precision, recall

    def get_precison_recall_results_interp(self) -> Tuple[np.array, np.array, np.array]:
        precision, recall, thresholds = precision_recall_curve(self.y_test, self.y_prob)
        precision_recall_df = pd.DataFrame({'precision': precision, 'recall': recall})
        precision_recall_df = precision_recall_df.sort_values('recall')
        recall_mean = np.linspace(0, 1, 100)
        interp_precision = np.interp(recall_mean, precision_recall_df['recall'], precision_recall_df['precision'])
        return interp_precision, recall_mean

    def get_visualization_util(self) -> VisualizationUtil:
        return VisualizationUtil()

    def display_graph_interp(self) -> VisualizationUtil:
        f, ax = plt.subplots(1, 3, figsize=(16, 5))
        plt.yticks(rotation=0)
        visualization_util = self.get_visualization_util()
        visualization_util.display_confusion_matrix(ax[0], self.get_confusion_matrix())
        visualization_util.display_roc_graph(ax[-2], *self.get_roc_results_interp())
        visualization_util.display_roc_threshold(ax[-2], *self.get_roc_results_interp())
        visualization_util.display_precision_recall(ax[-1], *self.get_precison_recall_results_interp())
        plt.show()
        return visualization_util

    def display_graph(self) -> VisualizationUtil:
        f, ax = plt.subplots(1, 3, figsize=(16, 5))
        plt.yticks(rotation=0)
        visualization_util = self.get_visualization_util()
        visualization_util.display_confusion_matrix(ax[0], self.get_confusion_matrix())
        visualization_util.display_roc_graph(ax[-2], *self.get_roc_results())
        visualization_util.display_roc_threshold(ax[-2], *self.get_roc_results())
        visualization_util.display_precision_recall(ax[-1], *self.get_precison_recall_results())
        plt.show()
        return visualization_util
