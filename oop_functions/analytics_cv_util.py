from __future__ import annotations

from typing import List

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import pickle

from oop_functions.analytics_util import AnalyticsUtil
from oop_functions.util_functions import print_df
from oop_functions.visualization_util import VisualizationUtil


class CvAnalyticsUtil:
    def __init__(self, analytics_utils: List[AnalyticsUtil], missing_df: pd.DataFrame, experiment_name: str = '') -> None:
        self.analytics_utils = analytics_utils
        self.missing_df = missing_df
        self.experiment_name = experiment_name
        self.k = self.get_num_folds()
        self.filter = None

    def set_filter(self, filter):
        self.filter = filter

    def get_label(self) -> str:
        return self.analytics_utils[0].data_util.label

    def get_classifier_type(self) -> str:
        return self.analytics_utils[0].classifier.__class__.__name__
    
    def get_num_folds(self) -> int:
        return len(self.analytics_utils)
    
    def get_file_suffix(self) -> str:
        return f'_for_experiment_{self.experiment_name}_{self.get_classifier_type()}_{self.get_label()}__{self.get_num_folds()}_trials'

    def update_thresholds(self):
        # TODO: update thresholds for each of the analytics utils
        pass

    def get_cv_report(self):
        cv_scores = []
        for k, analytics_util in enumerate(self.analytics_utils):
            report = analytics_util.get_report_generation_util_filtered(self.filter).generate_report().get_report()
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
            cm = analytics_util.get_report_generation_util_filtered(self.filter).get_confusion_matrix()
            confusion_matirices.append(cm)
        columns = confusion_matirices[0].columns
        index = confusion_matirices[0].index
        cv_cm = np.array(confusion_matirices)
        cv_cm = cv_cm.mean(axis=0)
        cv_cm = cv_cm.round(0)
        return pd.DataFrame(cv_cm, columns=columns, index=index).astype(np.int32)

    def roc_with_interval(self):
        fpr_mean = np.linspace(0, 1, 100)
        interp_tprs = []
        thresholds_list = []
        for k, analytics_util in enumerate(self.analytics_utils):
            fpr, tpr, thresholds = analytics_util.get_report_generation_util_filtered(self.filter).get_roc_results_interp()
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
            precision, recall = analytics_util.get_report_generation_util_filtered(self.filter).get_precision_recall_results_interp()
            interp_precision.append(precision)
        precision_mean = np.mean(interp_precision, axis=0)
        precision_std = np.std(interp_precision, axis=0)
        return precision_mean, recall_mean, precision_std * 1.96

    def display_graph(self) -> VisualizationUtil:
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
