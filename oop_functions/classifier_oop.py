from __future__ import annotations

import itertools
from typing import List, Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedKFold

from .util_functions import remove_featues_startswith, convert_numeric_to_float16, resample_max, select_numeric_columns

px_template = "simple_white"


class ImputerUtil:
    def __init__(self, impute_const_dict: Dict[str, float], impute_mean_cols: List[str] = [], impute_median_cols: List[str] = []) -> None:
        self.impute_const_dict = impute_const_dict
        self.impute_mean_cols = impute_mean_cols
        self.impute_median_cols = impute_median_cols
        
    def impute_data_const(self, train: pd.DataFrame, test: pd.DataFrame):
        const_val_cols = list(self.impute_const_dict.keys())
        # TODO: do I need this?
        # Only fill with const people who were screened
        # print(X_train[list(fill_const.keys())].describe().T)
        # X_train.loc[X_train['was_screened'] == 1] = X_train.loc[X_train['was_screened'] == 1].fillna(fill_const)
        # X_test.loc[X_test['was_screened'] == 1] = X_test.loc[X_test['was_screened'] == 1].fillna(fill_const)
        train = train.fillna(self.impute_const_dict)
        test = test.fillna(self.impute_const_dict)
        try: 
            train[const_val_cols] = train[const_val_cols].astype(np.int16)
            test[const_val_cols]  = test[const_val_cols].astype(np.int16)
        except:
            pass
        return train, test
    
    def impute_general(self, train: pd.DataFrame, test: pd.DataFrame, cols: List[str], strategy: str):
        if len(cols) > 0:
            imputer = SimpleImputer(missing_values=np.nan, strategy=strategy)
            imputer.fit(train[cols])
            train[cols] = imputer.transform(train[cols])
            test[cols] = imputer.transform(test[cols])
        return train, test

    def impute_data_mean(self, train: pd.DataFrame, test: pd.DataFrame):
        return self.impute_general(train, test, self.impute_mean_cols, 'mean')

    def impute_data_median(self, train: pd.DataFrame, test: pd.DataFrame):
        return self.impute_general(train, test, self.impute_median_cols, 'median')
    
    def impute_data(self, train: pd.DataFrame, test: pd.DataFrame):
        # keep in mind that it should stay float16 and not float64 use convert_numeric_to_float16(df)
        train, test = self.impute_data_const(train, test)
        train, test = self.impute_data_mean(train, test)
        train, test = self.impute_data_median(train, test)
        return convert_numeric_to_float16(train), convert_numeric_to_float16(test)

class ClassifierDataUtil:
    def __init__(self, label: str, imputer: ImputerUtil, id_col: str = 'plco_id', train_size: int = 10000, filtered_tests: dict = {}, debug: bool = False) -> None:
        # Train and test dfs contain split and imputed data, but still contain columns that have to be removed for training
        self.train_df = None
        self.test_df = None
        self.id_col = id_col
        self.label = label
        self.imputer = imputer
        self.debug = debug
        self.filtered_tests = filtered_tests
        self.stratify_tests_over_cols = list(self.filtered_tests.values())
        self.cols_to_remove = ['ovar_', 'cancer_', self.id_col, *self.stratify_tests_over_cols]
        self.train_size = train_size

    def copy(self) -> ClassifierDataUtil:
        return ClassifierDataUtil(
            label=self.label,
            imputer=self.imputer,
            id_col=self.id_col,
            train_size=self.train_size,
            filtered_tests=self.filtered_tests,
            debug=self.debug
        )
    
    def load_train_test_df(self, filename: str) -> ClassifierDataUtil:
        # TODO: be able to load and store the imputed data to be able to run experiments faster
        pass
    
    def store_train_test_df(self, filename: str) -> ClassifierDataUtil:
        # TODO: be able to load and store the imputed data to be able to run experiments faster
        pass
    
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
        y = df[self.label]
        X = df.drop([self.label], axis=1)
        return X, y
    
    def get_train_data(self) -> Tuple[pd.DataFrame, pd.Series]:
        df = remove_featues_startswith(self.train_df, self.cols_to_remove, [self.label], show_removed=False)
        return self.split_xy(df)
    
    def get_test_data(self) -> Tuple[pd.DataFrame, pd.Series]:
        df = remove_featues_startswith(self.test_df, self.cols_to_remove, [self.label], show_removed=False)
        return self.split_xy(df)
    
    def get_filtered_test_data(self) -> Tuple[pd.DataFrame, pd.Series, Tuple[str, List[int]]]:
        differentiated_test_sets = []
        filtered_on = list(itertools.chain.from_iterable([zip([key]*len(vals), vals) for key, vals in self.filtered_tests.items()]))
        for col, values in filtered_on:
            filtered_test = self.test_df[self.test_df[col].isin(values)]
            filtered_test = remove_featues_startswith(filtered_test, self.cols_to_remove, [self.label], show_removed=False)
            X_test_filtered, y_test_filtered = self.split_xy(filtered_test, self.label)
            differentiated_test_sets.append((X_test_filtered, y_test_filtered, (col, values)))
        return differentiated_test_sets

    def process_train_test_split(self, source_df: pd.DataFrame, train_ids: pd.Series, test_ids: pd.Series) -> ClassifierDataUtil:
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

    def split_kfold(self, num_folds: int = 10):        
        # One person should not appear in train and test data since there are duplicates of a person
        # we splits of data on person id and then oversample from that sample 
        # this line of code determines whether the model is leaking info or not
        unique_id_df = self.source_df[['plco_id', self.data_util.label]].drop_duplicates(subset='plco_id')

        # create list of lambdas for each fold
        strtfdKFold = StratifiedKFold(n_splits=num_folds)
        kfold = strtfdKFold.split(unique_id_df, unique_id_df[self.data_util.label])
        k_fold_lambdas = []
        for k, (train, test) in enumerate(kfold):
            train = unique_id_df.iloc[train, :]
            test = unique_id_df.iloc[test, :]
            k_fold_lambdas.append(lambda: self.data_util.copy().process_train_test_split(self.source_df, train[self.data_util.id_col], test[self.data_util.id_col]))

        return k_fold_lambdas


class ExperimentDataHelper:
    def __init__(self, source_df: pd.DataFrame, label: str, other_lebels: List[str], imputer_util: ImputerUtil = None, data_util: ClassifierDataUtil = None, train_test_split_util: TrainTestSplitUtil = None) -> None:
        self.source_df = source_df
        self.label = label
        self.other_lebels = other_lebels
        self.imputer_util = imputer_util
        self.data_util = data_util
        self.train_test_split_util = train_test_split_util
        self._process_source()

        if not imputer_util:
            self._init_imputer() 
        if not data_util:
            self._init_data_util()
        if not train_test_split_util:
            self._init_train_test_split_util()

    def _process_source(self) -> None:
        self.source_df = remove_featues_startswith(self.source_df, self.other_lebels, [self.label], show_removed=False).drop_duplicates()
        self.source_df = self.source_df[self.source_df[self.label].notnull()]

    def _init_imputer(self) -> None:
        pass

    def _init_data_util(self) -> None:
        self.data_util = ClassifierDataUtil(self.label, self.imputer_util)

    def _init_train_test_split_util(self) -> None:
        self.train_test_split_util = TrainTestSplitUtil(self.source_df, self.data_util, False)
        

class ExperimentDataHelper1(ExperimentDataHelper):
    def _init_imputer(self) -> None:
        impute_const_dict = {
            'numcyst': 0,
            'ovcyst_morph': 0,
            'ovcyst_outline': 0,
            'ovcyst_solid': 0,
            'ovcyst_sum': 0,
            'ovcyst_vol': 0,
            'numcyst': 0,
            'tvu_result': 1,
            'numcystl': 0,
            'numcystr': 0,
            'ovcyst_diaml': 0,
            'ovcyst_diamr': 0,
            'ovcyst_morphl': 0,
            'ovcyst_morphr': 0,
            'ovcyst_outlinel': 0,
            'ovcyst_outliner': 0,
            'ovcyst_solidl': 0,
            'ovcyst_solidr': 0,
            'ovcyst_suml': 0,
            'ovcyst_sumr': 0,
            'ovcyst_voll': 0,
            'ovcyst_volr': 0,
            'visboth': 0,
            'viseith': 0,
            'visl': 0,
            'visr': 0
        }
        numeric_columns = select_numeric_columns(self.source_df)
        numeric_columns = list(set(numeric_columns) - set(impute_const_dict.keys()))
        self.imputer_util = ImputerUtil(impute_const_dict, impute_mean_cols=numeric_columns, impute_median_cols=[])
