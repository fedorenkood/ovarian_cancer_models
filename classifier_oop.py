import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from collections import Counter
from imblearn.over_sampling import SMOTE
from itertools import product
from sklearn import datasets, linear_model, metrics, tree
from sklearn.ensemble import (AdaBoostClassifier, GradientBoostingClassifier,
                              RandomForestClassifier)
from sklearn.feature_selection import (SelectFromModel, SelectKBest, chi2,
                                       f_classif)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (ConfusionMatrixDisplay, PrecisionRecallDisplay,
                             accuracy_score,
                             auc, average_precision_score, classification_report,
                             confusion_matrix, f1_score, plot_precision_recall_curve,
                             plot_roc_curve, precision_recall_curve, precision_score,
                             recall_score, roc_auc_score, roc_curve)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_score, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils import resample
from tabulate import tabulate
from xgboost import XGBClassifier

import plotly.express as px
import seaborn as sns
import time
import warnings
import xgboost as xgb
from scipy.spatial import distance
import itertools
from typing import List, Dict, Tuple

px_template = "simple_white"

# https://scikit-learn.org/stable/modules/classes.html#module-sklearn.feature_selection

def summarize_features(df):
    # Summary of features
    # To understand whether we need to normalize features and we need to find unique value of each feature
    unique_values = {}
    unique_values['unique count'] = []
    # unique_values['values'] = []
    for col in df.columns:
        # unique_values['values'].append(sorted(df[col].unique()))
        unique_count = len(df[col].unique())
        unique_values['unique count'].append(unique_count)

    summary = pd.DataFrame(unique_values, index=df.columns).join(df.describe().T)
    return summary

def get_binary_features(df):
    cols_binary = []
    cols_quantitative = []
    for col in df.columns:
        unique_count = len(df[col].unique())
        if unique_count == 2:
            cols_binary.append(col)
        else:
            cols_quantitative.append(col)

    return cols_binary, cols_quantitative

def run_classifier(classifier, X_train, X_test, y_train, y_test, show_graph=True):
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)
    y_prob = classifier.predict_proba(X_test)[:,1]

    auc, accuracy, threshold, report = performance_analysis(y_pred, y_prob, y_test, show_graph=show_graph)
    
    print(f'Threshold: {threshold}')
    # Find prediction to the dataframe applying threshold
    y_pred_threathold = np.array(pd.Series(y_prob).map(lambda x: 1 if x > threshold else 0))
    performance_analysis(y_pred_threathold, y_prob, y_test, show_graph=show_graph)
    return classifier, auc, accuracy, threshold, report

def performance_analysis(y_pred, y_prob, y_test, show_graph=True):
    report = pd.DataFrame(classification_report(y_test, y_pred, output_dict=True)).transpose().iloc[0:2,:]
    auc = roc_auc_score(y_test, y_prob)
    accuracy = accuracy_score(y_test, y_pred)
    if show_graph:
        print(report)
        print(f'ROC AUC score: {auc}')
        print(f'Accuracy Score: {accuracy}')
        f, ax = plt.subplots(1, 3, figsize=(16, 5))
        plt.yticks(rotation = 0)
        display_confusion_matrix(ax[0], y_test, y_pred)

        threshold = display_auc(ax[-2], y_test, y_prob)
        display_precision_recall(ax[-1], y_prob, y_test)
        plt.show()
    report = report.drop(['support'], axis=1)
    report['class'] = report.index
    report['class'] = report['class'].astype('float')
    report['class'] = report['class'].astype('int')
    report['dummy'] = 0
    report = merge_df_into_features(report, on_col='dummy', make_unique_over_cols=['class'])
    report = remove_featues_startswith(report, ['class', 'dummy'], show_removed=False)
    report['accuracy'] = accuracy
    report['auc'] = auc
    return auc, accuracy, threshold, report


def get_accuracy(y_test, y_pred):
    return accuracy_score(y_test, y_pred)*100

def display_confusion_matrix(ax, y_test, y_pred):
    cm = pd.DataFrame(confusion_matrix(y_test, y_pred), columns=['Predicted Healthy', 'Predicted Cancer'], index=['Healthy', 'Cancer'])
    sns.heatmap(cm, annot = True, fmt = 'd', cbar = False, ax=ax)
    ax.set_title('Confusion Matrix')
    # ax.set_yticks(rotation = 0)

def display_auc(ax, y_test, y_pred):
    fpr, tpr, thresholds = roc_curve(y_test, y_pred)
    i = np.arange(len(tpr)) 
    roc = pd.DataFrame({'tf' : pd.Series(tpr-(1-fpr), index=i), 'threshold' : pd.Series(thresholds, index=i)})
    roc_t = roc.iloc[(roc.tf-0).abs().argsort()[:1]]
    optimal_idx = np.argmax(tpr - fpr)
    optimal_threshold = thresholds[optimal_idx]
    print(optimal_threshold)

    roc_auc = auc(fpr, tpr)
    ax.plot(fpr,tpr, color = '#b50000', label = 'AUC = %0.3f' % roc_auc)
    ax.plot([0, 1], [0, 1], linestyle = '-.', color = 'gray')
    ax.plot(fpr[optimal_idx],tpr[optimal_idx],'ro', label='Optimal point') 
    ax.set_ylabel('TP Rate')
    ax.set_xlabel('FP Rate')
    ax.set_title('ROC AUC Curve')
    ax.legend()
    # return list(roc_t['threshold']) 
    return optimal_threshold

def display_precision_recall(ax, y_pred, y_test):
    precision, recall, _ = precision_recall_curve(y_test, y_pred)
    disp = PrecisionRecallDisplay(precision=precision, recall=recall)
    disp.plot(ax=ax)
    ax.set_title('Precision-Recall Curve')

def mean_impute(df, columns):
    for col in columns:
        df[col] = df[col].fillna(df[col].mean())
    return df

def impute_with_val(df, columns, val):
    for col in columns:
        df[col] = df[col].fillna(val)
    return df

def select_numeric_columns(df):
    numeric_columns = df.select_dtypes(include=['float64','int64']).columns.tolist()
    return numeric_columns

def print_records_vs_unique(df, col, dataset_name, print_vals=True):
    # Get unique IDs
    if print_vals:
        print(f"Num of records in {dataset_name} dataset: {len(df)}")
        print(f"Num of unique {col} in {dataset_name} dataset: {len(df[col].unique())}")
    return len(df), len(df[col].unique())

def print_df(df):
    print(tabulate(df, headers='keys', tablefmt='psql'))

def print_records_vs_unique_for(df, col, dataset_name, on):
    print_records_vs_unique(df, col, dataset_name)
    # Look at unique IDs for each of the years of study for screen dataset
    for val in sorted(df[on].unique()):
        # Get unique IDs
        print_records_vs_unique(df[df[on] == val], col, f'{dataset_name}.{on}={val}')

def get_missing_values_cols(df):
    percent_missing = df.isnull().sum() * 100 / len(df)
    missing_value_df = pd.DataFrame({'column_name': df.columns,
                                    'num_missing': df.isnull().sum(),
                                    'num_present': len(df) - df.isnull().sum(),
                                    'percent_missing': percent_missing})
    missing_value_df.sort_values('percent_missing', inplace=True)
    # missing_value_df = missing_value_df[missing_value_df.percent_missing != 0]
    return missing_value_df

def get_unique_combinations(lists):
    return set(product(*lists))

def get_cols_missing_percentage(cutoff_percentage, df, name, show_missing=True):
    df_missing_value = get_missing_values_cols(df)
    df_missing_value.to_csv(f'./feature_selection/missing_percentage_{name}.csv', index=False)
    df_missing_value = df_missing_value[df_missing_value.percent_missing >= cutoff_percentage]
    if show_missing:
        print(f'{len(df_missing_value)} columns were over {cutoff_percentage} missing. This is the list of columns: {df_missing_value["column_name"].to_list()}')
    print(f'The table of features missing over {cutoff_percentage} percentage: ')
    if show_missing:
        print_df(df_missing_value)
    return df_missing_value

def drop_cols_missing_percentage(cutoff_percentage, df, name, show_missing=True):
    print(f'Removing features that are over {cutoff_percentage}% missing')
    df_missing_value = get_cols_missing_percentage(cutoff_percentage, df, name, show_missing=show_missing)
    return df.drop(df_missing_value['column_name'].to_list(), axis=1)
    
def remove_featues_startswith(df, prefixes, exclude=[], show_removed=True):
    for prefix in prefixes:
        remove_cols = []
        for col in df.columns:
            if col.startswith(prefix):
                remove_cols.append(col)
        if show_removed:
            print(f'Number of {prefix} cols: {len(remove_cols)}')
            print(remove_cols)
        remove_cols = list(set(remove_cols) - set(exclude))
        df = df.drop(remove_cols, axis=1)
    return df

def merge_df_into_features(sourse_df, on_col, make_unique_over_cols, join='outer'):
    unique_vals_list = [sorted(sourse_df[make_unique_over_col].unique()) for make_unique_over_col in make_unique_over_cols]
    unique_combinations_col_vals = get_unique_combinations(unique_vals_list)
    merged_df = None
    for unique_combination in unique_combinations_col_vals:
        filter = True
        col_suffix = ''
        for i in range(len(make_unique_over_cols)):
            variable_col = make_unique_over_cols[i]
            val = unique_combination[i]
            col_suffix += f'_{variable_col}_{val}'
            filter = filter & (sourse_df[variable_col] == val)
        df = sourse_df[filter]
        # Drop cols which we use to create unique features
        # df = df.drop(make_unique_over_cols, axis=1)
        # Create new col names
        cols = df.columns
        cols_dict = {}
        for col in cols:
            if col not in [on_col]:
                cols_dict[col] = f'{col}_{col_suffix}'
        df = df.rename(columns=cols_dict)
        # Get unique IDs
        if merged_df is None:
            merged_df = df
        else:
            merged_df = merged_df.merge(df, on=on_col, how=join)
    return merged_df



# Prints the table of missing values with columns for df filtered for each of the values of on_col
def df_missing_val_distribution_over_col(df, on_col, cutoff_percentage, title, show_missing=True):
    iterate_over_on_col = sorted(df[on_col].unique())
    suffixes=[]
    # since get_cols_missing_percentage will produce columns with the same names, we want to create suffixes to differentiate over on_col values
    for on_col_val in iterate_over_on_col:
        suffixes.append(f'_{on_col}_{on_col_val}')
    # We are going to merge all of the 
    df_missing = None
    for on_col_val in iterate_over_on_col:

        df_on_col = df[df[on_col] == on_col_val]

        df_missing_on_col = get_cols_missing_percentage(cutoff_percentage, df_on_col, f'{title} with {on_col}_{on_col_val}', show_missing=show_missing)
        
        if df_missing is not None:
            df_missing = df_missing.merge(df_missing_on_col, on='column_name', how='inner', suffixes=suffixes)
        else:
            df_missing = df_missing_on_col
    print_df(df_missing)


def df_filter_val_distribution(sourse_df, on_col, make_unique_over_cols, name, hist=True, cutoff_percentage=0, join='outer', suffixes=['_cancer', '_no_cancer']):
    unique_vals_list = [sorted(sourse_df[make_unique_over_col].unique()) for make_unique_over_col in make_unique_over_cols]
    unique_combinations_col_vals = get_unique_combinations(unique_vals_list)
    merged_df = None
    num_records, num_unique = print_records_vs_unique(sourse_df, 'plco_id', name)
    unique_combinations_col_vals = sorted(unique_combinations_col_vals)
    unique_records = [{} for _ in range(len(unique_combinations_col_vals)+1)]
    unique_records[0] = {
            'filtered_on': f'{name} total'
        }
    
    iterate_over_on_col = sorted(sourse_df[on_col].unique())
    for on_col_val in iterate_over_on_col:
        num_records, num_unique = print_records_vs_unique(sourse_df[sourse_df[on_col] == on_col_val], 'plco_id', f'{name} with {on_col}_{on_col_val}')
        unique_records[0][f'num_records_{on_col}:{on_col_val}'] = num_records
        unique_records[0][f'num_unique_id_{on_col}:{on_col_val}'] = num_unique
    print()
    print()
    for j in range(len(unique_combinations_col_vals)):
        unique_combination = unique_combinations_col_vals[j]
        filter = True
        filtered_on = ''
        for i in range(len(make_unique_over_cols)):
            variable_col = make_unique_over_cols[i]
            val = unique_combination[i]
            filter = filter & (sourse_df[variable_col] == val)
            filtered_on += f'{variable_col}: {val} '
        df = sourse_df[filter]
        if len(make_unique_over_cols) > 0:
            df = df.drop(make_unique_over_cols, axis=1)

        title = f'{name} filtered on {filtered_on}'
        print(title)

        df_missing = None
        unique_records[j+1]['filtered_on'] = title
        for on_col_val in iterate_over_on_col:
            df_on_col = df[df[on_col] == on_col_val]
            num_records, num_unique = print_records_vs_unique(df_on_col, 'plco_id', f'{title} with {on_col}_{on_col_val}')
            df_missing_on_col = get_cols_missing_percentage(cutoff_percentage, df_on_col, f'{title} with {on_col}_{on_col_val}')
            unique_records[j+1][f'num_records_{on_col}:{on_col_val}'] = num_records
            unique_records[j+1][f'num_unique_id_{on_col}:{on_col_val}'] = num_unique
            if hist:
                fig, ax = plt.subplots(1, 1, figsize=(20,15))
                df.hist(ax=ax, bins=30)
                plt.show()
        df_missing_val_distribution_over_col(df, on_col, cutoff_percentage, title, show_missing=False)
        df = impute_with_val(df, df.columns, -1)
        print('\n\n')
    print_df(pd.DataFrame(unique_records).sort_values('filtered_on'))


def df_filter_val_distribution_on_cancer(source_df, make_unique_over_cols, name, personal_data_cancer, personal_data_no_cancer, hist=False, cutoff_percentage=0):
    df_cancer = source_df[source_df['plco_id'].isin(personal_data_cancer['plco_id'])]
    df_cancer['cancer'] = 1
    df_no_cancer = source_df[source_df['plco_id'].isin(personal_data_no_cancer['plco_id'])]
    df_no_cancer['cancer'] = 0
    df = pd.concat([df_cancer, df_no_cancer], axis=0)
    df_filter_val_distribution(df, 'cancer', make_unique_over_cols, name, hist=hist, cutoff_percentage=cutoff_percentage)


class LabeledImmpute:
    def __init__(self) -> None:
        self.mean = None
        self.median = None

    def fit(self, df, label_col):
        self.mean = df.groupby([label_col], as_index=False).mean()
        self.median = df.groupby([label_col], as_index=False).median()

    def transform_base(self, df, label_col, values_df):
        df_list = []
        for label in df[label_col].unique():
            values_dict = values_df[values_df[label_col] == label].to_dict('records')[0]
            filtered_df = df[df[label_col] == label]
            filtered_df = filtered_df.fillna(values_dict)
            df_list.append(filtered_df)

        return pd.concat(df_list, axis=0)

    def transform_median(self, df, label_col):
        return self.transform_base(df, label_col, self.median)

    def transform_mean(self, df, label_col):
        return self.transform_base(df, label_col, self.mean)
    
    def transform(self, df, label_col, strategy):
        if strategy == 'mean':
            return self.transform_mean(df, label_col)
        return self.transform_median(df, label_col)



def convert_numeric_to_float16(df):
    numeric_cols = select_numeric_columns(df)
    df[numeric_cols] = df[numeric_cols].astype(np.float16)
    return df


def bucket_age(age, bucket_size):
    age_buckets = list(range(0, 101, bucket_size))
    for i in range(len(age_buckets) - 1):
        if age >= age_buckets[i] and age < age_buckets[i+1]:
            return age_buckets[i]
    return None


def merge_data_over_years(person_df, screen_df, abnorm_df, screen_join='left', abrorm_join='left'):
    on_col = 'ovar_cancer_years'
    # Select max data for each of the features in the abnorm_df, while varied over plco_id and study_yr
    abnorm_df = abnorm_df.groupby(['plco_id', 'study_yr'], as_index=False).max()
    df_list = []
    df_final = pd.DataFrame()
    for base_year in range(0, 19):
        df = person_df
        # individuals who got cancer before the beginning of this window should not be included in the current window
        df = df[df[on_col] >= base_year]
        # increment certain features by base year
        df['age'] = df['age'].apply(lambda age: bucket_age(age+base_year, 5))
        # If base year is 0 through 5 we have data on the scan performed on that year, so we can attach that data
        # we only use ca125ii_level features, but they are stored under different features for different years
        # we need to only use the ones for the appropriate year
        df = remove_featues_startswith(df, ['ca125ii_level'], [f'ca125ii_level{base_year}'], show_removed=False)
        if base_year <= 5: 
            df = df.rename({f'ca125ii_level{base_year}': 'ca125ii_level'}, axis=1)
            # print([col for col in df.columns if 'ca125ii_level' in col])
            df = df.merge(screen_df[screen_df['study_yr'] == base_year], how=screen_join)
        # If base year is 0 through 53 we have data on the abnormality on that year, so we can attach that data
        if base_year <= 3: 
            filtered_abnorm = abnorm_df[abnorm_df['study_yr'] == base_year]
            df = df.merge(filtered_abnorm, how=abrorm_join)
        # Assign new labels whether people will get cancer withing next 1, 3, 5, 10 years based on the current data
        for window_size in [1, 3, 5, 10]:
            if base_year + window_size >= 20:
                continue
            label_feature = f'cancer_in_next_{window_size}_years'
            df.loc[df[df[on_col] >= base_year + window_size].index, label_feature] = 0
            # df[label_feature] = 0
            index = df[df[on_col] < base_year + window_size].index
            df.loc[index, label_feature] = 1
        df_final = pd.concat([df_final, df])
        df_final = df_final.drop_duplicates()
    # Add a feature that says whether person was screened or not
    condition = df_final['plco_id'].isin(screen_df['plco_id'])
    df_final['was_screened'] = 0
    df_final.loc[condition, 'was_screened'] = 1
    return df_final

def resample_class(df, label, label_val, n_max_per_class, is_test=False, replace=True):
    df_majority = df[df[label] == label_val]
    df_minority = df[df[label] != label_val]

    # downsample df_majority class
    df_majority = resample(df_majority, 
                                    replace=False,     # sample with replacement
                                    n_samples=n_max_per_class,    # to match majority class
                                    # random_state=44
                                    ) 
    return pd.concat([df_majority, df_minority])

class ImputerUtil:
    def __init__(self, impute_const_dict: Dict[str, float], impute_mean_cols: List[str] = [], impute_median_cols: List[str] = []) -> None:
        self.impute_const_dict = impute_const_dict
        self.impute_mean_cols = impute_mean_cols
        self.impute_median_cols = impute_median_cols
        # TODO: this should be injected
        if self.impute_const_dict == None:
            self.impute_const_dict = {
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
        mean_imputer = SimpleImputer(missing_values=np.nan, strategy=strategy)
        mean_imputer.fit(train[cols])
        train[cols] = mean_imputer.transform(train[cols])
        test[cols] = mean_imputer.transform(test[cols])
        return train, test

    def impute_data_mean(self, train: pd.DataFrame, test: pd.DataFrame):
        return self.impute_general(train, test, self.impute_mean_cols, 'mean')

    def impute_data_median(self, train: pd.DataFrame, test: pd.DataFrame):
        return self.impute_general(train, test, self.impute_median_cols, 'median')
    
    def impute_data(self, train: pd.DataFrame, test: pd.DataFrame):
        # TODO: remove columns from here
        columns = train.columns
        # TODO: keep in mind that it should stay float16 and not float64
        train, test = self.impute_data_const(train, test)
        train, test = self.impute_data_mean(train, test)
        train, test = self.impute_data_median(train, test)
        return pd.DataFrame(train, columns=columns), pd.DataFrame(test, columns=columns)


class ClassifierDataUtil:
    def __init__(self, source_df: pd.DataFrame, train_ids: pd.Series, test_ids: pd.Series, label: str, imputer: ImputerUtil, train_size: int = 10000, filtered_tests: dict = {}, debug: bool = False) -> None:
        self.source_df = source_df
        self.train_ids = train_ids
        self.test_ids = test_ids
        # Train and test dfs contain split and imputed data, but still contain columns that have to be removed for training
        self.train_df = None
        self.test_df = None
        self.id_col = 'plco_id'
        self.label = label
        self.imputer = imputer
        self.debug = debug
        self.filtered_tests = filtered_tests
        self.stratify_tests_over_cols = list(self.filtered_tests.values())
        self.cols_to_remove = ['ovar_', 'cancer_', self.id_col, *self.stratify_tests_over_cols]
        self.train_size = train_size

    def get_id_from_index(self):
        # TODO: finish this one for diagnostics
        pass 

    def get_stats(self) -> None:
        if not self.train_df or not self.test_df:
            return
        y_train = self.train_df[self.label]
        y_test = self.test_df[self.label]
        print(f'Distribution of positive labels based on duplicate plco_id: {np.sum(y_test)/(np.sum(y_train) + np.sum(y_test))}')

    def split_xy(self, df) -> Tuple[pd.DataFrame, pd.DataFrame]:
        y = df[self.label]
        X = df.drop([self.label], axis=1)
        return X, y
    
    def get_train_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        df = remove_featues_startswith(self.train_df, self.cols_to_remove, [self.label], show_removed=False)
        return self.split_xy(df)
    
    def get_test_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        df = remove_featues_startswith(self.test_df, self.cols_to_remove, [self.label], show_removed=False)
        return self.split_xy(df)
    
    def get_filtered_test_data(self) -> Tuple[pd.DataFrame, pd.DataFrame, Tuple[str, List[int]]]:
        differentiated_test_sets = []
        filtered_on = list(itertools.chain.from_iterable([zip([key]*len(vals), vals) for key, vals in self.filtered_tests.items()]))
        for col, values in filtered_on:
            filtered_test = self.test_df[self.test_df[col].isin(values)]
            filtered_test = remove_featues_startswith(filtered_test, self.cols_to_remove, [self.label], show_removed=False)
            X_test_filtered, y_test_filtered = self.split_xy(filtered_test, self.label)
            differentiated_test_sets.append((X_test_filtered, y_test_filtered, (col, values)))
        return differentiated_test_sets

    def process_train_test_split(self):
        train = self.source_df[self.source_df[self.id_col].isin(self.train_ids)]
        test = self.source_df[self.source_df[self.id_col].isin(self.test_ids)]

        # Perform imputation before oversampling
        train, test = self.imputer.impute_data(train, test)

        # Perform oversamping and reshuffle
        train = resample_class(train, self.label, self.train_size, replace=True).sample(frac = 1)
        self.train_df = train.reset_index()
        self.test_df = test.reset_index()

        return self


class TrainTestSplitUtil:
    def __init__(self, source_df: pd.DataFrame, label: str, debug: bool = False) -> None:
        self.source_df = source_df
        self.id_col = 'plco_id'
        self.label = label
        self.debug = debug

    def split_kfold(self, strategy, n_max_per_class=10000, num_folds=10, differentiate_confusion_matrix_over=None):
        train_size = int(n_max_per_class * 0.8)
        test_size  = int(n_max_per_class * 0.2)
        train_fold_size  = int((num_folds-1) * train_size / num_folds)
        test_fold_size  = int(train_size / num_folds)

        # remove features starting with cancer so that we could drop labels that are nan (e.g. people get cancer later on)
        # TODO: this should happen before df is passed here
        source_df = remove_featues_startswith(source_df, ['cancer_'], [self.label], show_removed=False)
        source_df = source_df[source_df[self.label].notnull()]
        
        # One person should not appear in train and test data since there are duplicates of a person
        # we splits of data on person id and then oversample from that sample 
        # this line of code determines whether the model is leaking info or not
        unique_id_df = source_df[['plco_id', self.label]].drop_duplicates(subset='plco_id')
        X_train_unique, X_test_unique, y_train, y_test = train_test_split(unique_id_df, unique_id_df[self.label], test_size = 0.2)

        # Printing stats
        # print_records_vs_unique(X_train_unique, 'plco_id', f'Train set', print_vals=True)
        # print_records_vs_unique(X_test_unique, 'plco_id', f'Test set', print_vals=True)
        print(f'Distribution of labels based on unique plco_id: {np.sum(y_test)/(np.sum(y_train) + np.sum(y_test))}')
        
        train_test_lambda = lambda: self.process_train_test_split(source_df, X_train_unique, X_test_unique, self.label, train_size, test_size, strategy, differentiate_confusion_matrix_over=differentiate_confusion_matrix_over)
        # create list of lambdas for each fold
        # Cross validation: https://vitalflux.com/k-fold-cross-validation-python-example/
        strtfdKFold = StratifiedKFold(n_splits=num_folds)
        kfold = strtfdKFold.split(unique_id_df, unique_id_df[self.label])
        k_fold_lambdas = []
        for k, (train, test) in enumerate(kfold):
            train = unique_id_df.iloc[train, :]
            test = unique_id_df.iloc[test, :]
            # print_records_vs_unique(train, 'plco_id', f'Train cv fold {k}', print_vals=True)
            # print_records_vs_unique(test, 'plco_id', f'Test cv fold {k}', print_vals=True)
            k_fold_lambdas.append(lambda: self.process_train_test_split(source_df, train, test, self.label, train_fold_size, test_fold_size, strategy, stats=False))

        return train_test_lambda, k_fold_lambdas
