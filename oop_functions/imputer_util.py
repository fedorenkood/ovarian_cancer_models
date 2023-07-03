from __future__ import annotations

from typing import List, Dict

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer

from .util_functions import convert_numeric_to_float16


class ImputerUtil:
    def __init__(self, impute_const_dict: Dict[str, float], impute_mean_cols: List[str] = [],
                 impute_median_cols: List[str] = []) -> None:
        self.impute_const_dict = impute_const_dict
        self.impute_mean_cols = impute_mean_cols
        self.impute_median_cols = impute_median_cols
        self.imputer_mean = None
        self.imputer_median = None

    def copy(self):
        return ImputerUtil(
            self.impute_const_dict,
            self.impute_mean_cols,
            self.impute_median_cols
        )

    def impute_data_const(self, df: pd.DataFrame):
        const_val_cols = list(self.impute_const_dict.keys())
        # TODO: do I need this?
        # Only fill with const people who were screened
        # print(X_train[list(fill_const.keys())].describe().T)
        # X_train.loc[X_train['was_screened'] == 1] = X_train.loc[X_train['was_screened'] == 1].fillna(fill_const)
        # X_test.loc[X_test['was_screened'] == 1] = X_test.loc[X_test['was_screened'] == 1].fillna(fill_const)
        df = df.fillna(self.impute_const_dict)
        try:
            df[const_val_cols] = df[const_val_cols].astype(np.int16)
        except:
            pass
        return df
    
    def fit_imputer(self, train: pd.DataFrame, cols: List[str], strategy: str):
        if len(cols) == 0: 
            return 
        if strategy == 'mean':
            self.imputer_mean.fit(train[cols])
        if strategy == 'median':
            self.imputer_median.fit(train[cols])

    def imputer_transform_general(self, df: pd.DataFrame, cols: List[str], strategy: str):
        if len(cols) == 0: 
            return df
        if strategy == 'mean':
            df[cols] = self.imputer_mean.transform(df[cols])
        if strategy == 'median':
            df[cols] = self.imputer_median.transform(df[cols])
        return df
    
    def imputer_transform(self, df: pd.DataFrame):
        df = self.impute_data_const(df)
        df = self.imputer_transform_general(df, self.impute_mean_cols, 'mean')
        df = self.imputer_transform_general(df, self.impute_median_cols, 'median')
        return df
    
    def imputer_fit(self, train: pd.DataFrame, test: pd.DataFrame):
        # keep in mind that it should stay float16 and not float64 use convert_numeric_to_float16(df)
        self.imputer_mean = SimpleImputer(missing_values=np.nan, strategy='mean')
        self.imputer_median = SimpleImputer(missing_values=np.nan, strategy='median')
        self.fit_imputer(train, self.impute_mean_cols, 'mean')
        self.fit_imputer(train, self.impute_median_cols, 'median')
        return train, test

    def impute_data(self, train: pd.DataFrame, test: pd.DataFrame):
        train, test = self.imputer_fit(train, test)
        train = self.imputer_transform(train)
        test = self.imputer_transform(test)
        return convert_numeric_to_float16(train), convert_numeric_to_float16(test)
