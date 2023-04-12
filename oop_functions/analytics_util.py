from __future__ import annotations

from typing import Tuple, Callable

import numpy as np
import pandas as pd
import xgboost as xgb
from matplotlib import pyplot as plt
from sklearn import tree
from sklearn.base import clone

from .classifier_data_util import ClassifierDataUtil
from .report_util import GenerateReportUtil


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

    def plot_save_tree(self, plot_tree: bool = False, filepath: str = None) -> AnalyticsUtil:
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
    def plot_save_tree(self, plot_tree: bool = False, filepath: str = None) -> DesicionTreeAnalyticsUtil:
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
    def plot_save_tree(self, plot_tree: bool = False, filepath: str = None) -> XgbAnalyticsUtil:
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
