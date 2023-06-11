from __future__ import annotations

from typing import List, Tuple, Callable

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
import pickle

from .imputer_util import ImputerUtil
from .util_functions import remove_featues_startswith, resample_max, convert_numeric_to_float16


class ClassifierDataUtil:
    def __init__(self, label: str, imputer: ImputerUtil, id_col: str = 'plco_id', train_size: int = 10000, stratify_tests_over_cols: list = [], debug: bool = False) -> None:
        # Train and test dfs contain split and imputed data, but still contain columns that have to be removed for training
        self.train_df = None
        self.test_df = None
        self.id_col = id_col
        self.label = label
        self.imputer = imputer
        self.debug = debug
        self.stratify_tests_over_cols = stratify_tests_over_cols
        self.cols_to_remove = ['ovar_', 'cancer_', self.id_col, 'index', *stratify_tests_over_cols]
        self.train_size = train_size

    def copy(self) -> ClassifierDataUtil:
        return ClassifierDataUtil(
            label=self.label,
            imputer=self.imputer.copy(),
            id_col=self.id_col,
            train_size=self.train_size,
            stratify_tests_over_cols=self.stratify_tests_over_cols,
            debug=self.debug
        )
    
    def load_train_test_df(self, filesuffix: str) -> ClassifierDataUtil:
        # be able to load and store the imputed data to be able to run experiments faster
        self.train_df = pd.read_csv(f'./imputed_data/train_{filesuffix}.csv')
        self.train_df = convert_numeric_to_float16(self.train_df)
        self.test_df = pd.read_csv(f'./imputed_data/test_{filesuffix}.csv')
        self.test_df = convert_numeric_to_float16(self.test_df)
        return self
    
    def store_train_test_df(self, filesuffix: str) -> ClassifierDataUtil:
        # be able to load and store the imputed data to be able to run experiments faster
        self.train_df.to_csv(f'./imputed_data/train_{filesuffix}.csv', index=False)
        self.test_df.to_csv(f'./imputed_data/test_{filesuffix}.csv', index=False)
        return self
    
    def check_if_data_util_initialized(self) -> None:
        if self.train_df is None or self.test_df is None:
            raise Exception("Data Util was not initialized")

    def get_record_from_train_index(self, index: int) -> str:
        return self.train_df.loc[index, :]

    def get_record_from_test_index(self, index: int) -> str:
        return self.test_df.loc[index, :]

    def get_id_from_train_index(self, index: int) -> str:
        return self.get_record_from_train_index(index)[self.id_col]

    def get_id_from_test_index(self, index: int) -> str:
        return self.get_record_from_test_index(index)[self.id_col]

    def get_stats(self) -> None:
        self.check_if_data_util_initialized()
        y_train = self.train_df[self.label]
        y_test = self.test_df[self.label]
        print(f'Distribution of positive labels based on duplicate plco_id: {np.sum(y_test)/(np.sum(y_train) + np.sum(y_test))}')

    def split_xy(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        df = remove_featues_startswith(df, self.cols_to_remove, [self.label, 'ovar_result'], show_removed=False)
        y = df[self.label]
        X = df.drop([self.label], axis=1)
        return X, y
    
    def get_train_data(self) -> Tuple[pd.DataFrame, pd.Series]:
        return self.split_xy(self.train_df)
    
    def get_test_data(self) -> Tuple[pd.DataFrame, pd.Series]:
        return self.split_xy(self.test_df)
    
    def get_feature_names(self) -> List[str]:
        X, y = self.split_xy(pd.DataFrame([], columns=self.test_df.columns))
        return list(X.columns)
    
    def get_filtered_test_data(self, filter: Callable[[pd.DataFrame], pd.DataFrame]) -> Tuple[pd.DataFrame, pd.Series]:
        filtered_test = filter(self.test_df)
        filtered_test = remove_featues_startswith(filtered_test, self.cols_to_remove, [self.label, 'ovar_result'], show_removed=False)
        X_test_filtered, y_test_filtered = self.split_xy(filtered_test)
        return X_test_filtered, y_test_filtered

    def process_train_test_split(self, source_df: pd.DataFrame, train_ids: pd.Series, test_ids: pd.Series) -> ClassifierDataUtil:
        if self.imputer is None:
            return self
        train = source_df[source_df[self.id_col].isin(train_ids)]
        test = source_df[source_df[self.id_col].isin(test_ids)]

        # Perform imputation before oversampling
        train, test = self.imputer.impute_data(train, test)

        # Perform oversamping and reshuffle
        train = resample_max(train, self.label, self.train_size).sample(frac = 1)
        # TODO: if memory becomes tight, only store index to id tuples
        self.train_df, self.test_df = train, test
        return self


class TrainTestSplitUtil:
    def __init__(self, source_df: pd.DataFrame, data_util: ClassifierDataUtil, debug: bool = False) -> None:
        self.source_df = source_df
        self.data_util = data_util
        self.debug = debug

    def split_kfold(self, num_folds: int = 10, max_test: int = None):        
        # One person should not appear in train and test data since there are duplicates of a person
        # we splits of data on person id and then oversample from that sample 
        # this line of code determines whether the model is leaking info or not
        unique_id_df = self.source_df[[self.data_util.id_col, self.data_util.label]].drop_duplicates(subset=self.data_util.id_col)

        # create list of lambdas for each fold
        strtfdKFold = StratifiedKFold(n_splits=num_folds)
        kfold = strtfdKFold.split(unique_id_df, unique_id_df[self.data_util.label])
        k_fold_lambdas = []
        for k, (train, test) in enumerate(kfold):
            if max_test is not None and k > max_test:
                break
            train = unique_id_df.iloc[train, :]
            test = unique_id_df.iloc[test, :]
            k_fold_lambdas.append(self.data_util.copy().process_train_test_split(self.source_df, train[self.data_util.id_col], test[self.data_util.id_col]))

        return k_fold_lambdas
