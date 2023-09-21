from __future__ import annotations

from typing import List
import math

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import pickle
from sklearn.metrics import roc_curve, precision_recall_curve

from oop_functions.analytics_util import AnalyticsUtil
from oop_functions.util_functions import print_df
from oop_functions.visualization_util import VisualizationUtil
from oop_functions.report_util import GenerateReportUtil


class CvAnalyticsUtil:
    def __init__(self, analytics_utils: List[AnalyticsUtil], missing_df: pd.DataFrame, experiment_name: str = '') -> None:
        self.analytics_utils = analytics_utils
        self.missing_df = missing_df
        self.experiment_name = experiment_name
        self.k = self.get_num_folds()
        self.filter = None
        self.threshold = None

    def set_filter(self, filter):
        self.filter = filter

    def get_label(self) -> str:
        return self.analytics_utils[0].data_util.label

    def get_classifier_type(self) -> str:
        return self.analytics_utils[0].classifier.__class__.__name__
    
    def get_num_folds(self) -> int:
        return len(self.analytics_utils)
    
    def get_features(self):
        return self.analytics_utils[0].data_util.test_df.columns
    
    def get_file_suffix(self) -> str:
        return f'_for_experiment_{self.experiment_name}_{self.get_classifier_type()}_{self.get_label()}__{self.get_num_folds()}_trials'

    def update_thresholds(self, threshold: int):
        # TODO: update thresholds for each of the analytics utils
        self.threshold = threshold

    def get_dataset_with_predictions(self, filter = None):
        id_and_confidence = []
        for analytics_util in self.analytics_utils:
            X_test = analytics_util.data_util.test_df
            X_test_mismatch = X_test.copy()
            y_pred, y_prob = analytics_util.get_predictions() 
            X_test_mismatch[f'{self.get_label()}_pred'] = y_pred
            X_test_mismatch[f'{self.get_label()}_prob'] = y_prob
            id_and_confidence.append(X_test_mismatch)
        full_dataset = pd.concat(id_and_confidence)
        if filter:
            full_dataset = filter(full_dataset)
        return full_dataset

    def merge_in_dataset(self, dataset):
        cols = self.get_features()
        full_dataset_original = self.get_dataset_with_predictions()[cols]
        full_dataset = dataset[cols]
        full_dataset = full_dataset[~full_dataset['index'].isin(full_dataset_original['index'])]
        full_dataset = pd.concat([full_dataset, full_dataset_original])
        # make sure that the original records stay the same and we just add new ones 249 new ones to be exact
        # Add those records to the test datasets of the single label dataset and test its validity
        for analytics_util in self.analytics_utils:
            idx = analytics_util.data_util.test_df['plco_id'].to_list()
            # print(analytics_util.data_util.test_df.shape)
            # print((full_dataset[full_dataset['plco_id'].isin(idx)].shape))
            analytics_util.data_util.test_df = full_dataset[full_dataset['plco_id'].isin(idx)]
            analytics_util.data_util.test_df = analytics_util.data_util.imputer.imputer_transform(analytics_util.data_util.test_df)

        full_dataset_single_new = self.get_dataset_with_predictions()
        print(f"Added new records: {len(full_dataset_single_new) - len(full_dataset_original)}")


    def get_cv_report(self):
        cv_scores = []
        for k, analytics_util in enumerate(self.analytics_utils):
            # TODO: cleanup and generalize
            try:
                report_generation_util = analytics_util.get_report_generation_util_filtered(self.filter)
            except Exception as e:
                # print(f'Filter resulted in error. i.e. no records with such filter')
                continue
            if self.threshold:
                report_generation_util.apply_threshold(self.threshold)
            report = report_generation_util.generate_report().get_report()
            cv_scores.append(report)
        cv_scores = pd.concat(cv_scores)
        cv_scores = cv_scores.reset_index()
        cv_scores = cv_scores.drop('index', axis=1)
        measures_df = cv_scores.describe().T[['mean', 'std', 'min', 'max']]
        print('\n\nCross-Validation measures:')
        print_df(measures_df)
        return cv_scores, measures_df

    def get_confusion_matrix(self):
        confusion_matirices = []
        for k, analytics_util in enumerate(self.analytics_utils):
            try:
                report_generation_util = analytics_util.get_report_generation_util_filtered(self.filter)
            except Exception as e:
                # print(f'Filter resulted in error. i.e. no records with such filter')
                continue
            if self.threshold:
                report_generation_util.apply_threshold(self.threshold)
            cm = report_generation_util.get_confusion_matrix()
            confusion_matirices.append(cm)
        columns = confusion_matirices[0].columns
        index = confusion_matirices[0].index
        cv_cm = np.array(confusion_matirices)
        cv_cm = cv_cm.sum(axis=0)
        cv_cm = cv_cm.round(0)
        return pd.DataFrame(cv_cm, columns=columns, index=index).astype(np.int32)
    
    def combined_predictions(self):
        y_test_all = []
        y_pred_all = []
        y_prob_all = []
        for k, analytics_util in enumerate(self.analytics_utils):
            try:
                report_generation_util = analytics_util.get_report_generation_util_filtered(self.filter)
            except Exception as e:
                # print(f'Filter resulted in error. i.e. no records with such filter')
                continue
            if self.threshold:
                report_generation_util.apply_threshold(self.threshold)
            y_test, y_pred, y_prob = report_generation_util.get_predictions()
            y_test_all.append(y_test)
            y_pred_all.append(y_pred)
            y_prob_all.append(y_prob)
        y_test_all = np.concatenate(y_test_all, axis=0)
        y_pred_all = np.concatenate(y_pred_all, axis=0)
        y_prob_all = np.concatenate(y_prob_all, axis=0)
        return y_test_all, y_pred_all, y_prob_all
    
    def get_optimal_operating_point(self):
        y_test_all, y_pred_all, y_prob_all = self.combined_predictions()
        report_util = GenerateReportUtil(y_test_all, y_pred_all, y_prob_all)
        return report_util.get_roc_threshold()

    def roc_curve(self):
        y_test_all, y_pred_all, y_prob_all = self.combined_predictions()
        fpr, tpr, thresholds = roc_curve(y_test_all, y_prob_all)
        return fpr, tpr, thresholds

    def precision_recall(self):
        y_test_all, y_pred_all, y_prob_all = self.combined_predictions()
        precision, recall, thresholds = precision_recall_curve(y_test_all, y_prob_all)
        return precision, recall, thresholds

    def roc_with_interval(self):
        fpr_mean = np.linspace(0, 1, 100)
        interp_tprs = []
        thresholds_list = []
        for k, analytics_util in enumerate(self.analytics_utils):
            try:
                report_generation_util = analytics_util.get_report_generation_util_filtered(self.filter)
            except Exception as e:
                # print(f'Filter resulted in error. i.e. no records with such filter')
                continue
            if self.threshold:
                report_generation_util.apply_threshold(self.threshold)
            fpr, tpr, thresholds = report_generation_util.get_roc_results_interp()
            interp_tprs.append(tpr)
            thresholds_list.append(thresholds)
        tpr_mean = np.mean(interp_tprs, axis=0)
        tpr_mean[-1] = 1.0
        tpr_std = np.std(interp_tprs, axis=0)
        thresholds_mean = np.mean(thresholds_list, axis=0)
        return fpr_mean, tpr_mean, thresholds_mean, tpr_std * 1.96

    def precision_recall_with_interval(self):
        recall_mean = np.linspace(0, 1, 100)
        interp_precision = []
        for k, analytics_util in enumerate(self.analytics_utils):
            try:
                report_generation_util = analytics_util.get_report_generation_util_filtered(self.filter)
            except Exception as e:
                # print(f'Filter resulted in error. i.e. no records with such filter')
                continue
            if self.threshold:
                report_generation_util.apply_threshold(self.threshold)
            precision, recall = report_generation_util.get_precision_recall_results_interp()
            interp_precision.append(precision)
        precision_mean = np.mean(interp_precision, axis=0)
        precision_std = np.std(interp_precision, axis=0)
        return precision_mean, recall_mean, precision_std * 1.96

    def display_graph(self) -> VisualizationUtil:
        f, ax = plt.subplots(1, 3, figsize=(16, 5))
        plt.yticks(rotation=0)
        visualization_util = VisualizationUtil()
        visualization_util.display_confusion_matrix(ax[0], self.get_confusion_matrix())
        visualization_util.display_roc_graph(ax[-2], *self.roc_curve())
        # visualization_util.display_roc_threshold(ax[-2], *self.roc_with_interval())
        precision, recall, thresholds = self.precision_recall()
        visualization_util.display_precision_recall(ax[-1], precision, recall)
        plt.show()
        return visualization_util

    def display_graph_interval(self) -> VisualizationUtil:
        f, ax = plt.subplots(1, 3, figsize=(16, 5))
        plt.yticks(rotation=0)
        visualization_util = VisualizationUtil()
        visualization_util.display_confusion_matrix(ax[0], self.get_confusion_matrix())
        visualization_util.display_roc_graph(ax[-2], *self.roc_with_interval())
        # visualization_util.display_roc_threshold(ax[-2], *self.roc_with_interval())
        visualization_util.display_precision_recall(ax[-1], *self.precision_recall_with_interval())
        plt.show()
        return visualization_util

    def get_cv_feature_selection(self) -> pd.DataFrame:
        pass

    def store_cv_results(self) -> None:
        # I assume that all of the analytics_utils have the same classifier type
        cv_scores, measures_df = self.get_cv_report()
        cv_scores.to_csv(f'./cv_scores/cv_scores_{self.get_file_suffix()}.csv')
        measures_df.to_csv(f'./cv_scores/cv_stats_{self.get_file_suffix()}.csv')

    def store_cv_analytics_utils(self, filesuffix: str) -> None:
        for k, analytics_util in enumerate(self.analytics_utils):
            analytics_util.store_analytics_util(f'{filesuffix}_{k+1}_fold')
            
        analytics_utils = self.analytics_utils
        self.analytics_utils = None
        pickle.dump(self, open(f'./stored_classes/cv_analytics_util/{filesuffix}.sav', 'wb'))
        self.analytics_utils = analytics_utils

    @classmethod
    def load_cv_analytics_utils(cls, filesuffix: str) -> CvAnalyticsUtil:
        cv_analytics_util = pickle.load(open(f'./stored_classes/cv_analytics_util/{filesuffix}.sav', 'rb'))
        analytics_utils = []
        for k in range(cv_analytics_util.k):
            analytics_util = AnalyticsUtil.load_analytics_util(f'{filesuffix}_{k+1}_fold')
            analytics_utils.append(analytics_util)
        cv_analytics_util.analytics_utils = analytics_utils
        return cv_analytics_util


class FeatureImportanceCvAnalyticsUtil(CvAnalyticsUtil):
    def get_cv_feature_selection_ver2(self) -> pd.DataFrame:
        df_feature_importance_tree = None
        dfs = []
        for k, analytics_util in enumerate(self.analytics_utils):
            fn = analytics_util.data_util.get_feature_names()
            top_feature_stats, feature_importances = analytics_util.feature_selection()
            feature_importances = feature_importances[feature_importances['importance'] > 0]
            dfs.append(feature_importances)
            if df_feature_importance_tree is not None:
                df_feature_importance_tree = df_feature_importance_tree.merge(feature_importances, on='column_name',
                                                                              how='outer', suffixes=[f'_tiral_{k}',
                                                                                                     f'_tiral_{k + 1}'])
            else:
                df_feature_importance_tree = feature_importances

        print(f"Number of dfs: {len(dfs)}")
        # Concatenate the DataFrames into a single DataFrame
        combined_df = pd.concat(dfs, ignore_index=True)

        # Group by 'column_name' and aggregate the 'importance' and 'std' columns
        agg_result = combined_df.groupby('column_name').agg({
            'importance': 'mean',              # Calculate the mean of 'importance'
            'std': lambda x: math.sqrt(sum(x**2)),  # Combine 'std' using the root of sum of squares
        }).reset_index()
        agg_result.columns = ['column_name', 'mean', 'std']
        agg_result.sort_values('mean', ascending=False, inplace=True)
        # Mean of feature importance over trials
        df_feature_importance_mean = df_feature_importance_tree.drop('column_name', axis=1)
        df_feature_importance_mean = df_feature_importance_mean.T
        df_feature_importance_mean.columns = df_feature_importance_tree['column_name']
        df_feature_importance_mean = df_feature_importance_mean.astype('float')
        df_feature_importance_mean_describe = df_feature_importance_mean.describe().T
        df_feature_importance_mean_describe.sort_values('mean', ascending=False, inplace=True)
        df_feature_importance_mean_describe = df_feature_importance_mean_describe[['count']]
        # print(df_feature_importance_mean_describe.columns)
        # df_feature_importance_mean_describe = df_feature_importance_mean_describe[['count', 'mean']]
        # print_df(df_feature_importance_mean_describe)
        df_feature_importance_mean_describe = df_feature_importance_mean_describe.merge(self.missing_df,
                                                                                        on='column_name')
        agg_result = agg_result.merge(df_feature_importance_mean_describe, on='column_name')
        print_df(agg_result)
        return agg_result
    
    def get_cv_feature_selection(self) -> pd.DataFrame:
        df_feature_importance_tree = None
        for k, analytics_util in enumerate(self.analytics_utils):
            fn = analytics_util.data_util.get_feature_names()
            top_feature_stats, feature_importances = analytics_util.feature_selection()
            feature_importances = feature_importances[feature_importances['importance'] > 0]
            if df_feature_importance_tree is not None:
                df_feature_importance_tree = df_feature_importance_tree.merge(feature_importances, on='column_name',
                                                                              how='outer', suffixes=[f'_tiral_{k}',
                                                                                                     f'_tiral_{k + 1}'])
            else:
                df_feature_importance_tree = feature_importances
        # Mean of feature importance over trials
        df_feature_importance_mean = df_feature_importance_tree.drop('column_name', axis=1)
        df_feature_importance_mean = df_feature_importance_mean.T
        df_feature_importance_mean.columns = df_feature_importance_tree['column_name']
        df_feature_importance_mean = df_feature_importance_mean.astype('float')
        df_feature_importance_mean_describe = df_feature_importance_mean.describe().T
        df_feature_importance_mean_describe.sort_values('mean', ascending=False, inplace=True)
        # print(df_feature_importance_mean_describe.columns)
        df_feature_importance_mean_describe = df_feature_importance_mean_describe[['count', 'mean']]
        # print_df(df_feature_importance_mean_describe)
        df_feature_importance_mean_describe = df_feature_importance_mean_describe.merge(self.missing_df,
                                                                                        on='column_name')
        return df_feature_importance_mean_describe

    def store_cv_results(self) -> None:
        super(FeatureImportanceCvAnalyticsUtil, self).store_cv_results()
        df_feature_importance_mean_describe = self.get_cv_feature_selection()
        df_feature_importance_mean_describe.to_csv(
            f'./feature_importance/feature_importance_mean_{self.get_file_suffix()}.csv')
