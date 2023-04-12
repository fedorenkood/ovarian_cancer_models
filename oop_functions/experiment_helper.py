from __future__ import annotations

from typing import List

import pandas as pd

from .classifier_data_util import ClassifierDataUtil, TrainTestSplitUtil
from .imputer_util import ImputerUtil
from .util_functions import remove_featues_startswith, select_numeric_columns


class ExperimentDataHelper:
    def __init__(self, source_df: pd.DataFrame, label: str, other_lebels: List[str], imputer_util: ImputerUtil = None,
                 data_util: ClassifierDataUtil = None, train_test_split_util: TrainTestSplitUtil = None) -> None:
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
        self.source_df = remove_featues_startswith(self.source_df, self.other_lebels, [self.label],
                                                   show_removed=False).drop_duplicates()
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

    def _process_source(self) -> None:
        super(ExperimentDataHelper1, self)._process_source()
        # drop non-cancer records without screen records
        condition = (self.source_df['was_screened'] == 1) | (self.source_df['ovar_cancer'] == 1)
        self.source_df = self.source_df[condition]
