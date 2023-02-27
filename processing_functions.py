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

def scale_features(df):
    sc = StandardScaler()
    df_scaled = df.drop('Cancer_binary', axis=1)
    df_scaled = sc.fit_transform(df_scaled)
    df_scaled = pd.DataFrame(df_scaled, columns=df.columns[1:])
    df_scaled.insert(0, 'Cancer_binary', df['Cancer_binary'].to_numpy())
    return df_scaled

def scaling_tranform(X_train, X_test):
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)
    return X_train, X_test

def run_classifier(classifier, X_train, X_test, y_train, y_test, show_graph=True):
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)
    y_prob = classifier.predict_proba(X_test)[:,1]
    auc, accuracy, precision, recall, f1 = performance_analysis(y_pred, y_prob, y_test, show_graph=show_graph)
    return auc, accuracy, precision, recall, f1

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

        display_auc(ax[1], y_test, y_prob)

        display_precision_recall(ax[2], y_prob, y_test)
        plt.show()
    return auc, accuracy, precision_score(y_test, y_pred), recall_score(y_test, y_pred), f1_score(y_test, y_pred)


def get_accuracy(y_test, y_pred):
    return accuracy_score(y_test, y_pred)*100

def display_confusion_matrix(ax, y_test, y_pred):
    cm = pd.DataFrame(confusion_matrix(y_test, y_pred), columns=['Predicted Healthy', 'Predicted Cancer'], index=['Healthy', 'Cancer'])
    sns.heatmap(cm, annot = True, fmt = 'd', cbar = False, ax=ax)
    ax.set_title('Confusion Matrix')
    # ax.set_yticks(rotation = 0)

def display_auc(ax, y_test, y_pred):
    fp_rate, tp_rate, thresholds = roc_curve(y_test, y_pred)
    roc_auc = auc(fp_rate, tp_rate)
    ax.plot(fp_rate,tp_rate, color = '#b50000', label = 'AUC = %0.3f' % roc_auc)
    ax.plot([0, 1], [0, 1], linestyle = '-.', color = 'gray')
    ax.set_ylabel('TP Rate')
    ax.set_xlabel('FP Rate')
    ax.set_title('ROC AUC Curve')
    ax.legend()

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


def resample_max(df, label, n_max_per_class, replace=True):

    df_majority = df[df[label] ==0]
    df_minority = df[df[label] ==1]

    # downsample df_majority class
    df_majority_downsampled = resample(df_majority, 
                                    replace=False,     # sample with replacement
                                    n_samples=n_max_per_class,    # to match majority class
                                    # random_state=44
                                    ) 

    # Upsample minority class
    df_minority_upsampled = resample(df_minority, 
                                    replace=replace,     # sample with replacement
                                    n_samples=n_max_per_class,    # to match majority class
                                    # random_state=44
                                    ) 

    # Combine majority class with upsampled minority class
    df_sampled = pd.concat([df_majority_downsampled, df_minority_upsampled])

    return df_sampled



def process_and_impute_for_label(source_df, label, strategy, n_max_per_class=10000):
    resampled_df = resample_max(source_df, label, n_max_per_class, replace=True)
    x = resampled_df
    y = x[label]
    x = x.drop(['ovar_cancer_years', 'plco_id'], axis=1)
    # TODO: One person should not appear in train and test data since there are duplicates?
    # TODO: do splits of data on person id and then oversample from that space 
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = 0.2)

    mean_imputer = SimpleImputer(missing_values=np.nan, strategy=strategy)
    mean_imputer.fit(X_train)
    X_test = mean_imputer.transform(X_test)
    X_train = pd.DataFrame(X_train, columns=x.columns)
    X_test = pd.DataFrame(X_test, columns=x.columns)

    labeled_impute = LabeledImmpute()
    labeled_impute.fit(X_train, label)
    # X_train = labeled_impute.transform(X_train, label, strategy)
    X_train = mean_imputer.transform(X_train)
    X_train = pd.DataFrame(X_train, columns=x.columns)
    try:
        X_train = X_train.drop([label], axis=1)
        X_test = X_test.drop([label], axis=1)
        X_train = remove_featues_startswith(X_train, ['ovar_', 'cancer_'])
        X_test = remove_featues_startswith(X_test, ['ovar_', 'cancer_'])
    except:
        pass

    return X_train, X_test, y_train, y_test
