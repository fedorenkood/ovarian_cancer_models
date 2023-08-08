from __future__ import annotations

from typing import Tuple, Callable

import numpy as np
import pandas as pd
import xgboost as xgb
from matplotlib import pyplot as plt
from scipy.spatial import distance
from sklearn import tree
from sklearn.base import clone
from sklearn.preprocessing import StandardScaler
import pickle

from .classifier_data_util import ClassifierDataUtil
from .report_util import GenerateReportUtil
from .util_functions import get_nearest_neighbors


class AnalyticsUtil:
    def __init__(self, classifier, data_util: ClassifierDataUtil) -> None:
        self.classifier = clone(classifier)
        self.data_util = data_util

    def store_analytics_util(self, filesuffix: str) -> None:
        self.data_util.store_train_test_df(filesuffix)
        self.data_util.store_imputer(filesuffix)
        train_df = self.data_util.train_df
        test_df  = self.data_util.test_df
        self.data_util.train_df = None
        self.data_util.test_df  = None
        # store class in a pickle
        pickle.dump(self, open(f'./stored_classes/analytics_util/{filesuffix}.sav', 'wb'))
        self.data_util.train_df = train_df
        self.data_util.test_df  = test_df

    @staticmethod
    def load_analytics_util(filesuffix: str) -> AnalyticsUtil:
        # load trained classifier from file
        analytics_util = pickle.load(open(f'./stored_classes/analytics_util/{filesuffix}.sav', 'rb'))
        analytics_util.data_util.load_train_test_df(filesuffix)
        analytics_util.data_util.load_imputer(filesuffix)
        return analytics_util

    def fit(self) -> AnalyticsUtil:
        self.data_util.check_if_data_util_initialized()

        X_train, y_train = self.data_util.get_train_data()
        self.classifier.fit(X_train, y_train)
        return self

    def get_predictions_general(self, X_test: pd.DataFrame) -> Tuple[np.array, np.array]:
        y_pred = self.classifier.predict(X_test)
        y_prob = self.classifier.predict_proba(X_test)[:, 1]
        return y_pred, y_prob

    def get_predictions(self) -> Tuple[np.array, np.array]:
        X_test, y_test = self.data_util.get_test_data()
        return self.get_predictions_general(X_test)
    
    def get_predictions_filtered(self, filter: Callable[[pd.DataFrame], pd.DataFrame]) -> GenerateReportUtil:
        if filter is None: 
            return self.get_report_generation_util()
        X_test, y_test = self.data_util.get_filtered_test_data(filter)
        y_pred, y_prob = self.get_predictions_general(X_test) 
        return y_pred, y_prob
    
    def get_report_generation_util(self) -> GenerateReportUtil:
        X_test, y_test = self.data_util.get_test_data()
        y_pred, y_prob = self.get_predictions() 
        return GenerateReportUtil(y_test, y_pred, y_prob)
    
    def get_report_generation_util_filtered(self, filter: Callable[[pd.DataFrame], pd.DataFrame]) -> GenerateReportUtil:
        if filter is None: 
            return self.get_report_generation_util()
        X_test, y_test = self.data_util.get_filtered_test_data(filter)
        y_pred, y_prob = self.get_predictions_general(X_test) 
        return GenerateReportUtil(y_test, y_pred, y_prob)

    def plot_save_tree(self, plot_tree: bool = False, filepath: str = None) -> AnalyticsUtil:
        return self

    def feature_selection(self):
        pass
    
    def get_high_confidence_errors(self):
        label = self.data_util.label
        # Insert predicted class and its likelihood
        X_test, y_test = self.data_util.get_test_data()
        X_test = X_test.copy()
        y_pred, y_prob = self.get_predictions() 
        X_test_mismatch = X_test.copy()
        X_test_mismatch[label] = y_test
        X_test_mismatch[f'{label}_pred'] = y_pred
        X_test_mismatch[f'{label}_prob'] = y_prob
        X_test_mismatch = X_test_mismatch.drop_duplicates()
        X_test_mismatch = X_test_mismatch[X_test_mismatch[label] != X_test_mismatch[f'{label}_pred']]
        return X_test_mismatch
    
    def get_mismatches_neightbors(self, label_val: int = 0, top_n: int = 5):
        label = self.data_util.label
        X_train, y_train = self.data_util.get_train_data()
        X_test, y_test = self.data_util.get_test_data()
        X_test_mismatch = self.get_high_confidence_errors()

        # X_test_high_conf = X_test_mismatch[(X_test_mismatch[f'{label}_prob'] < 0.2) | (X_test_mismatch[f'{label}_prob'] > 0.8)]
        X_test_high_conf = X_test_mismatch
        X_test_high_conf = X_test_high_conf[X_test_high_conf[f'{label}_pred'] == label_val]
        if X_test_high_conf.shape[0] == 0:
            return []
        
        # Select 5 nearest neightbors 
        X_train[label] = y_train
        X_train_filtered = X_train[X_train[label] == label_val].drop(label, axis=1)
        # X_test_high_conf = X_test.loc[X_test_high_conf.index, :]
        # Calculated euclidean distances
        distances, indices = get_nearest_neighbors(X_test.loc[X_test_high_conf.index, :], X_train_filtered, top_n)
        fp_mismatches = []
        X_train[f'{label}_pred'] = -1
        X_train[f'{label}_prob'] = -1
        X_train = X_train.drop_duplicates()
        # print_df(X_train)
        for i in range(len(X_test_high_conf)):
            idx = indices[i]
            missed_record = X_test_high_conf.iloc[[i], :]
            missed_record['distance'] = 0
            close_records = X_train.loc[idx, :]
            close_records['distance'] = distances[i]
            fp_mismatches.append((missed_record, close_records))
        return fp_mismatches


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
        tree_depth = self.get_max_depth()
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


class DecisionTreeAnalyticsUtil(TreeAnalyticsUtil):
    def plot_save_tree(self, plot_tree: bool = False, filepath: str = None) -> DecisionTreeAnalyticsUtil:
        if plot_tree or filepath is not None:
            print('Plots tree')
            cn=['no cancer', 'cancer']
            fig, axes = plt.subplots(nrows = 1,ncols = 1, dpi=3000)
            tree.plot_tree(self.classifier,
                        max_depth=4,
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
