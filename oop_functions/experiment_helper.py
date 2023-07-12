from __future__ import annotations

from typing import List

import pandas as pd
import numpy as np

from .classifier_data_util import ClassifierDataUtil, TrainTestSplitUtil
from .imputer_util import ImputerUtil
from .util_functions import remove_featues_startswith, select_numeric_columns, get_cols_missing_percentage

# TODO: these have changed
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
       'viseith', 'numcystl', 'numcystr', 'numcyst', 'plco_id', 'ovar_days',
       'ca125ii_level_binary']

screen_abnorm_data_cols = ['study_yr', 'solid', 'sepst', 'cyst', 'cystw', 'echo', 'maxdi', 'volum',
       'plco_id']

screened_cols = screen_data_cols + screen_abnorm_data_cols + ['ca125ii', 'ca125_result']


screen_data_cols_fill_last = ['detl_p', 'detr_p', 'lvol_p', 'rvol_p', 'lvol_q', 'rvol_q',
       'lantero_p', 'lantero_q', 'llong_p', 'llong_q', 'ltran_p', 'ltran_q',
       'rantero_p', 'rantero_q', 'rlong_p', 'rlong_q', 'rtran_p', 'rtran_q',
       'tvu_ref', 'phycons', 'tvu_result', 'ovar_result',
       'ovcyst_solidr', 'ovcyst_outliner', 'ovcyst_solidl', 'ovcyst_outlinel',
       'ovcyst_solid', 'ovcyst_outline', 'ovcyst_diamr', 'ovcyst_diaml',
       'ovcyst_diam', 'ovcyst_volr', 'ovcyst_voll', 'ovcyst_vol',
       'ovcyst_morphr', 'ovcyst_morphl', 'ovcyst_morph', 'ovcyst_sumr',
       'ovcyst_suml', 'ovcyst_sum', 'ovary_diam', 'ovary_diamr', 'ovary_diaml',
       'ovary_volr', 'ovary_voll', 'ovary_vol', 'visl', 'visr', 'visboth',
       'viseith', 'numcystl', 'numcystr', 'numcyst',
    #    'ca125_result', 
       'ca125ii_level_binary']

screen_abnorm_data_fill_last = ['solid', 'sepst', 'cyst', 'cystw', 'echo', 'maxdi', 'volum']

class ExperimentDataHelper:
    def __init__(self, source_df: pd.DataFrame, label: str, other_lebels: List[str], train_size: int = 10000, imputer_util: ImputerUtil = None,
                 data_util: ClassifierDataUtil = None, train_test_split_util: TrainTestSplitUtil = None) -> None:
        self.source_df = source_df
        self.label = label
        self.other_lebels = other_lebels
        self.imputer_util = imputer_util
        self.data_util = data_util
        self.train_test_split_util = train_test_split_util
        self.stratify_over_cols = self.set_stratify_over_cols_default()
        self.train_size = train_size
        self.missing_df = None
        self._process_source()
        self.source_df = self.source_df.drop_duplicates(list(set(self.source_df.columns) - set(['index', *self.stratify_over_cols])))
        self._propagate_values()

        # Propagate previous values
        # Impute data for patient with the most recent value for a feature if at all present
        # I removed ca125ii feature from being propagated because I noticed that doing that increased accuracy.

        

        if not imputer_util:
            self._init_imputer()
        if not data_util:
            self._init_data_util()
        if not train_test_split_util:
            self._init_train_test_split_util()

    @staticmethod
    def get_name() -> str:
        return 'experiment'
    
    def set_stratify_over_cols_default(self) -> List[str]:
        return ['was_screened', 'ovar_histtype', 'study_yr', 'ovar_observe_year', 'ovar_cancer_years']
    
    def set_train_size_to_val(self, val) -> None:
        self.train_size = val
        self.data_util.train_size = val
    
    def set_train_size_to_max(self, k: int) -> None:
        self.set_train_size_to_val(int(self.source_df[self.source_df[self.label] == 0][self.data_util.id_col].nunique()/10*9))
        
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

    # def _use_most_recent(self, df, keep_last_on_col, col_id, on_cols):
    #     impute_in_order = sorted(df[keep_last_on_col].unique(), reverse=True)
    #     for col in on_cols:
    #         # print(impute_values)
    #         # TODO: fix this one only fill out the values that are past the study year
    #         for unique_year in impute_in_order:
    #             sorted_df = df[df[col].notnull()]
    #             sorted_df = sorted_df[sorted_df[keep_last_on_col] == unique_year]
    #             # sorted_df = sorted_df.sort_values(by=keep_last_on_col)
    #             impute_values = sorted_df.drop_duplicates(col_id, keep='last')
    #             impute_values = impute_values[[col_id, col]]
    #             df_for_impute = df
    #             df_for_impute = df_for_impute[(df_for_impute[col].isnull()) & (df_for_impute['ovar_observe_year'] > unique_year)][[col_id, 'ovar_observe_year', col]]
    #             index = df_for_impute.index
    #             df.loc[index, [col_id, 'ovar_observe_year', col]] = df_for_impute.set_index(col_id).combine_first(impute_values.set_index(col_id)).reset_index()
    #     return df
    
    def _use_most_recent(self, df, keep_last_on_col, col_id, on_cols):
        for col in on_cols:
            if col in df.columns:
                sorted_df = df[df[col].notnull()]
                sorted_df = sorted_df.sort_values(by=keep_last_on_col)
                impute_values = sorted_df.drop_duplicates(col_id, keep='last')
                impute_values = impute_values[[col_id, col]]
                df = df.set_index(col_id).combine_first(impute_values.set_index(col_id)).reset_index()
        return df

    def _propagate_values(self):
        # Custom ca125 propagation
        # if 'ca125ii_level' in self.source_df.columns:
        #     did_not_get_cancer_after_elevated_index = []
        #     for year in range(0,6):
        #         filtered = self.source_df[(self.source_df['ovar_observe_year'] == year) & (self.source_df['ca125ii_level'] >= 35) & (self.source_df['ovar_cancer_years'] > year + 1)]
        #         index = self.source_df[(self.source_df['ovar_observe_year'] == year + 1) 
        #                             & (self.source_df['ca125ii_level'].isna()) 
        #                             & (self.source_df['plco_id'].isin(filtered['plco_id']))].index
        #         did_not_get_cancer_after_elevated_index.extend(index)
        #     self.source_df.loc[did_not_get_cancer_after_elevated_index, 'ca125ii_level'] = 34
        if 'ca125ii_level_binary' in self.source_df.columns:
            did_not_get_cancer_after_elevated_index = []
            for year in range(0,6):
                filtered = self.source_df[(self.source_df['ovar_observe_year'] == year) & (self.source_df['ca125ii_level_binary'] >= 2) & (self.source_df['ovar_cancer_years'] > year + 1)]
                index = self.source_df[(self.source_df['ovar_observe_year'] == year + 1) 
                                    & (self.source_df['ca125ii_level_binary'].isna()) 
                                    & (self.source_df['plco_id'].isin(filtered['plco_id']))].index
                did_not_get_cancer_after_elevated_index.extend(index)
            self.source_df.loc[did_not_get_cancer_after_elevated_index, 'ca125ii_level_binary'] = 1
        # Other values propagation
        original_missing = get_cols_missing_percentage(0, self.source_df, 'merged_df', False)[['column_name', 'percent_missing']]
        self.source_df['study_yr'] = self.source_df['study_yr'].fillna(-1)
        self.source_df = self._use_most_recent(self.source_df, 'study_yr', 'plco_id', screen_data_cols_fill_last + screen_abnorm_data_fill_last)
        # self.source_df = self.source_df.drop('study_yr', axis=1)
        after_prop_missing = get_cols_missing_percentage(0, self.source_df, 'last_propagated_df', False)[['column_name', 'percent_missing']]
        self.missing_df = original_missing.merge(after_prop_missing, suffixes=['_before_propagation', '_after_propagation'], on='column_name')


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
            'visr': 0,
            'ovar_histtype': -1,
            'ph_any_bq': 9,
            'ph_ovar_bq': 9,
            'ph_any_not_ovar_bq': 9,
        }
        numeric_columns = select_numeric_columns(self.source_df)
        numeric_columns = list(set(numeric_columns) - set(impute_const_dict.keys()))
        self.imputer_util = ImputerUtil(impute_const_dict, impute_mean_cols=numeric_columns, impute_median_cols=[])
    

class ExperimentDataHelperWithImputerSingleLabel(ExperimentDataHelperWithImputer):
    def _process_source(self) -> None:
        super(ExperimentDataHelperWithImputerSingleLabel, self)._process_source()
        # experiment where participants cannot be classified by two different labels i.e. first classfied as not having cancer and then classified as having cancer
        idx = self.source_df[self.source_df[self.label] == 1]['plco_id'].to_list()
        condition = ~( (self.source_df[self.label] == 0) & self.source_df['plco_id'].isin(idx)) 
        self.source_df = self.source_df[ condition]


class ExperimentDataHelperSingleLabelScreenedOrCancer(ExperimentDataHelperWithImputerSingleLabel):
    @staticmethod
    def get_name() -> str:
        return 'participants_screened_or_cancer_single_label'
    
    def _process_source(self) -> None:
        super(ExperimentDataHelperSingleLabelScreenedOrCancer, self)._process_source()
        # drop non-cancer records without screen records
        condition = (self.source_df['was_screened'] == 1) | (self.source_df['ovar_cancer'] == 1)
        self.source_df = self.source_df[condition]

class ExperimentDataHelperSingleLabelScreened(ExperimentDataHelperWithImputerSingleLabel):
    @staticmethod
    def get_name() -> str:
        return 'participants_screened_single_label'
    
    def _process_source(self) -> None:
        super(ExperimentDataHelperSingleLabelScreened, self)._process_source()
        # drop non-cancer records without screen records
        condition = (self.source_df['was_screened'] == 1)
        self.source_df = self.source_df[condition]
        # TODO: why did I decide to keep this?
        self.source_df['ca125ii_level_binary'] = np.nan
        self.source_df.loc[self.source_df['ca125ii_level'] < 35, 'ca125ii_level_binary'] = 1
        self.source_df.loc[self.source_df['ca125ii_level'] >= 35 , 'ca125ii_level_binary'] = 2

class ExperimentDataHelperSingleLabelScreenedFirst5(ExperimentDataHelperSingleLabelScreened):
    @staticmethod
    def get_name() -> str:
        return 'participants_screened_single_first_5'
    
    def _process_source(self) -> None:
        condition = ((self.source_df['ovar_observe_year'] <= 5)) 
        self.source_df = self.source_df[condition]
        super(ExperimentDataHelperSingleLabelScreenedFirst5, self)._process_source()
        
class ExperimentDataHelperSingleLabelNotScreenedCols(ExperimentDataHelperWithImputerSingleLabel):
    @staticmethod
    def get_name() -> str:
        return 'not_screened_cols_single_label'
    
    def _process_source(self) -> None:
        super(ExperimentDataHelperSingleLabelNotScreenedCols, self)._process_source()
        self.source_df = remove_featues_startswith(self.source_df, screened_cols, exclude=['plco_id', 'index', *self.stratify_over_cols], show_removed=False)
        
class ExperimentDataHelperSingleLabelScreenedCols(ExperimentDataHelperSingleLabelScreened):
    @staticmethod
    def get_name() -> str:
        return 'screened_cols_single_label'
    
    def _process_source(self) -> None:
        super(ExperimentDataHelperSingleLabelScreenedCols, self)._process_source()
        keep_cols_screen = []
        for col in screened_cols + self.stratify_over_cols + [self.label, 'index']:
            if col in self.source_df.columns:
                keep_cols_screen.append(col)
        self.source_df = self.source_df[list(set(keep_cols_screen))]
        
class ExperimentDataHelperSingleLabelAll(ExperimentDataHelperWithImputerSingleLabel):
    @staticmethod
    def get_name() -> str:
        return 'participants_all_single_label'
    
    def _process_source(self) -> None:
        super(ExperimentDataHelperSingleLabelAll, self)._process_source()


class ExperimentDataHelperScreenedOrCancer(ExperimentDataHelperWithImputer):
    @staticmethod
    def get_name() -> str:
        return 'participants_screened_or_cancer'
    
    def _process_source(self) -> None:
        super(ExperimentDataHelperScreenedOrCancer, self)._process_source()
        # drop non-cancer records without screen records
        condition = (self.source_df['was_screened'] == 1) | (self.source_df['ovar_cancer'] == 1)
        self.source_df = self.source_df[condition]

class ExperimentDataHelperScreened(ExperimentDataHelperWithImputer):
    @staticmethod
    def get_name() -> str:
        return 'participants_screened'
    
    def _process_source(self) -> None:
        super(ExperimentDataHelperScreened, self)._process_source()
        # drop non-cancer records without screen records
        condition = (self.source_df['was_screened'] == 1)
        self.source_df = self.source_df[condition]
        # TODO: why did I decide to keep this?
        self.source_df['ca125ii_level_binary'] = np.nan
        self.source_df.loc[self.source_df['ca125ii_level'] < 35, 'ca125ii_level_binary'] = 1
        self.source_df.loc[self.source_df['ca125ii_level'] >= 35 , 'ca125ii_level_binary'] = 2


class ExperimentDataHelperScreenedFirst5(ExperimentDataHelperScreened):
    @staticmethod
    def get_name() -> str:
        return 'participants_screened_first_5'
    
    def _process_source(self) -> None:
        condition = (self.source_df['ovar_observe_year'] <= 5)
        self.source_df = self.source_df[condition]
        super(ExperimentDataHelperScreenedFirst5, self)._process_source()


class ExperimentDataHelperScreenedFirst5ca125AndBinary(ExperimentDataHelperScreenedFirst5):
    @staticmethod
    def get_name() -> str:
        return 'participants_screened_first_5_ca125_and_binary'
    
    def _process_source(self) -> None:
        super(ExperimentDataHelperScreenedFirst5ca125AndBinary, self)._process_source()
        self.source_df['ca125ii_level_binary'] = np.nan
        self.source_df.loc[self.source_df['ca125ii_level'] < 35, 'ca125ii_level_binary'] = 1
        self.source_df.loc[self.source_df['ca125ii_level'] >= 35 , 'ca125ii_level_binary'] = 2


class ExperimentDataHelperScreenedFirst5ca125Binary(ExperimentDataHelperScreenedFirst5ca125AndBinary):
    @staticmethod
    def get_name() -> str:
        return 'participants_screened_first_5_ca125_binary'
    
    def _process_source(self) -> None:
        super(ExperimentDataHelperScreenedFirst5ca125Binary, self)._process_source()
        self.source_df = self.source_df.drop('ca125ii_level', axis=1)


class ExperimentDataHelperScreenedFirst5ca125AndBinaryNoResult(ExperimentDataHelperScreenedFirst5ca125AndBinary):
    @staticmethod
    def get_name() -> str:
        return 'participants_screened_first_5_ca125_and_binary_no_result'
    
    def _process_source(self) -> None:
        super(ExperimentDataHelperScreenedFirst5ca125AndBinaryNoResult, self)._process_source()
        self.source_df = self.source_df.drop('ca125_result', axis=1)
        
class ExperimentDataHelperNotScreenedCols(ExperimentDataHelperWithImputer):
    @staticmethod
    def get_name() -> str:
        return 'not_screened_cols'
    
    def _process_source(self) -> None:
        super(ExperimentDataHelperNotScreenedCols, self)._process_source()
        self.source_df = remove_featues_startswith(self.source_df, screened_cols, exclude=['plco_id', 'index', *self.stratify_over_cols], show_removed=False)


class ExperimentDataHelperNotScreenedColsFirst5(ExperimentDataHelperNotScreenedCols):
    @staticmethod
    def get_name() -> str:
        return 'not_screened_cols_first_5'
    
    def _process_source(self) -> None:
        super(ExperimentDataHelperNotScreenedColsFirst5, self)._process_source()
        condition = (self.source_df['ovar_observe_year'] <= 5)
        self.source_df = self.source_df[condition]

     
class ExperimentDataHelperScreenedCols(ExperimentDataHelperScreened):
    @staticmethod
    def get_name() -> str:
        return 'screened_cols'
    
    def _process_source(self) -> None:
        super(ExperimentDataHelperScreenedCols, self)._process_source()
        keep_cols_screen = []
        for col in screened_cols + self.stratify_over_cols + [self.label, 'index']:
            if col in self.source_df.columns:
                keep_cols_screen.append(col)
        self.source_df = self.source_df[list(set(keep_cols_screen))]


class ExperimentDataHelperScreenedColsFirst5(ExperimentDataHelperScreenedCols):
    @staticmethod
    def get_name() -> str:
        return 'participants_screened_cols_first_5'
    
    def _process_source(self) -> None:
        super(ExperimentDataHelperScreenedColsFirst5, self)._process_source()
        condition = (self.source_df['ovar_observe_year'] <= 5)
        self.source_df = self.source_df[condition]
        
class ExperimentDataHelperAll(ExperimentDataHelperWithImputer):
    @staticmethod
    def get_name() -> str:
        return 'participants_all'
    
    def _process_source(self) -> None:
        super(ExperimentDataHelperAll, self)._process_source()
