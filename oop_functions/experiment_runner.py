from typing import Type

import pandas as pd
from sklearn import clone

from oop_functions.analytics_cv_util import FeatureImportanceCvAnalyticsUtil, CvAnalyticsUtil
from oop_functions.analytics_util import AnalyticsUtil, DecisionTreeAnalyticsUtil, XgbAnalyticsUtil
from oop_functions.experiment_helper import ExperimentDataHelper


class ExperimentRunner:
    def __init__(self, 
                 classifier, 
                 experiment_data_helper: ExperimentDataHelper, 
                 num_folds: int = 10, 
                 test_n_folds: int = 1, 
                 n_repeats: int = 1) -> None:
        self.classifier = clone(classifier) 
        self.experiment_data_helper = experiment_data_helper
        # Instead of running all folds just run several folds so that it is easier to bench test
        self.num_folds = num_folds
        # TODO: RepeatedStratifiedKFold instead of StratifiedKFold to run experiments
        self.n_repeats = n_repeats
        self.test_n_folds = test_n_folds
        if self.test_n_folds is None:
            self.test_n_folds = self.num_folds * self.n_repeats

    def get_analytics_util(self) -> Type[AnalyticsUtil]:
        if self.classifier.__class__.__name__ == 'DecisionTreeClassifier':
            return DecisionTreeAnalyticsUtil
        if self.classifier.__class__.__name__ == 'XGBClassifier':
            return XgbAnalyticsUtil
        return AnalyticsUtil

    def get_cv_analytics_util(self) -> Type[CvAnalyticsUtil]:
        if self.classifier.__class__.__name__ == 'DecisionTreeClassifier':
            return FeatureImportanceCvAnalyticsUtil
        if self.classifier.__class__.__name__ == 'XGBClassifier':
            return FeatureImportanceCvAnalyticsUtil
        return CvAnalyticsUtil
    
    def run_experiment(self) -> CvAnalyticsUtil:
        data_util_lambdas = self.experiment_data_helper.train_test_split_util.split_kfold(self.num_folds, self.test_n_folds)
        analytics_utils = []
        for i in range(self.test_n_folds):
            data_util = data_util_lambdas[i]
            data_util = data_util.impute_transform()
            analytics_util = self.get_analytics_util()(self.classifier, data_util)
            analytics_util = analytics_util.fit()
            analytics_utils.append(analytics_util)
        return self.get_cv_analytics_util()(analytics_utils, self.experiment_data_helper.missing_df, self.experiment_data_helper.get_name())
