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
            test[const_val_cols] = test[const_val_cols].astype(np.int16)
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
