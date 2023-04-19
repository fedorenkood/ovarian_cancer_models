from __future__ import annotations

from typing import List

import pandas as pd

from .classifier_data_util import ClassifierDataUtil, TrainTestSplitUtil
from .imputer_util import ImputerUtil
from .util_functions import remove_featues_startswith, select_numeric_columns

screen_data_cols = ['study_yr', 'detl_p', 'detr_p', 'lvol_p', 'rvol_p', 'lvol_q', 'rvol_q',
       'lantero_p', 'lantero_q', 'llong_p', 'llong_q', 'ltran_p', 'ltran_q',
       'rantero_p', 'rantero_q', 'rlong_p', 'rlong_q', 'rtran_p', 'rtran_q',
       'tvu_ref', 'phycons', 'tvu_result', 'ca125_result', 'ovar_result',
       'ovcyst_solidr', 'ovcyst_outliner', 'ovcyst_solidl', 'ovcyst_outlinel',
       'ovcyst_solid', 'ovcyst_outline', 'ovcyst_diamr', 'ovcyst_diaml',
       'ovcyst_diam', 'ovcyst_volr', 'ovcyst_voll', 'ovcyst_vol',
       'ovcyst_morphr', 'ovcyst_morphl', 'ovcyst_morph', 'ovcyst_sumr',
       'ovcyst_suml', 'ovcyst_sum', 'ovary_diam', 'ovary_diamr', 'ovary_diaml',
       'ovary_volr', 'ovary_voll', 'ovary_vol', 'visl', 'visr', 'visboth',
       'viseith', 'numcystl', 'numcystr', 'numcyst', 'plco_id', 'ovar_days']

screen_abnorm_data_cols = ['study_yr', 'solid', 'sepst', 'cyst', 'cystw', 'echo', 'maxdi', 'volum',
       'plco_id']

screened_cols = screen_data_cols + screen_abnorm_data_cols + ['ca125ii', 'ca125_result']

class ExperimentDataHelper:
    def __init__(self, source_df: pd.DataFrame, label: str, other_lebels: List[str], imputer_util: ImputerUtil = None,
                 data_util: ClassifierDataUtil = None, train_test_split_util: TrainTestSplitUtil = None) -> None:
        self.source_df = source_df
        self.label = label
        self.other_lebels = other_lebels
        self.imputer_util = imputer_util
        self.data_util = data_util
        self.train_test_split_util = train_test_split_util
        self.stratify_over_cols = self.set_stratify_over_cols_default()
        self.train_size = 10000
        self._process_source()
        

        if not imputer_util:
            self._init_imputer()
        if not data_util:
            self._init_data_util()
        if not train_test_split_util:
            self._init_train_test_split_util()

    def get_name(self) -> str:
        return 'experiment'
    
    def set_stratify_over_cols_default(self) -> List[str]:
        return ['was_screened', 'ovar_histtype']
    
    def set_train_size_to_val(self, val) -> None:
        self.train_size = val
        self.data_util.train_size = val
    
    def set_train_size_to_max(self) -> None:
        self.set_train_size_to_val(len(self.data_util.train_df))
        
    def _process_source(self) -> None:
        self.source_df = remove_featues_startswith(self.source_df, self.other_lebels, [self.label],
                                                   show_removed=False).drop_duplicates()
        self.source_df = self.source_df[self.source_df[self.label].notnull()]

    def _init_imputer(self) -> None:
        pass

    def _init_data_util(self) -> None:
        self.data_util = ClassifierDataUtil(self.label, self.imputer_util, train_size = self.train_size, stratify_tests_over_cols = self.stratify_over_cols)

    def _init_train_test_split_util(self) -> None:
        self.train_test_split_util = TrainTestSplitUtil(self.source_df, self.data_util, False)


class ExperimentDataHelperWithImputer(ExperimentDataHelper):
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

class ExperimentDataHelperScreenedOrCancer(ExperimentDataHelperWithImputer):
    def get_name(self) -> str:
        return 'participants_screened_or_cancer'
    
    def _process_source(self) -> None:
        super(ExperimentDataHelperScreenedOrCancer, self)._process_source()
        # drop non-cancer records without screen records
        condition = (self.source_df['was_screened'] == 1) | (self.source_df['ovar_cancer'] == 1)
        self.source_df = self.source_df[condition]
        
class ExperimentDataHelperNotScreenedCols(ExperimentDataHelperWithImputer):
    def get_name(self) -> str:
        return 'not_screened_cols'
    
    def _process_source(self) -> None:
        super(ExperimentDataHelperNotScreenedCols, self)._process_source()
        self.source_df = remove_featues_startswith(self.source_df, screened_cols, exclude=['plco_id', *self.stratify_over_cols], show_removed=False)
        
class ExperimentDataHelperScreenedCols(ExperimentDataHelperWithImputer):
    def get_name(self) -> str:
        return 'screened_cols'
    
    def _process_source(self) -> None:
        super(ExperimentDataHelperScreenedCols, self)._process_source()
        # drop non screen records
        condition = (self.source_df['was_screened'] == 1)
        self.source_df = self.source_df[condition]
        keep_cols_screen = []
        for col in screened_cols + self.stratify_over_cols + [self.label]:
            if col in self.source_df.columns:
                keep_cols_screen.append(col)
        self.source_df = self.source_df[list(set(keep_cols_screen))]
        
class ExperimentDataHelperAll(ExperimentDataHelperWithImputer):
    def get_name(self) -> str:
        return 'participants_all'
    
    def _process_source(self) -> None:
        super(ExperimentDataHelperAll, self)._process_source()
