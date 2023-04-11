from itertools import product

import numpy as np
import pandas as pd
from sklearn.utils import resample
from tabulate import tabulate


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


def print_df(df):
    print(tabulate(df, headers='keys', tablefmt='psql'))


def get_unique_combinations(lists):
    return set(product(*lists))


def get_missing_values_cols(df):
    percent_missing = df.isnull().sum() * 100 / len(df)
    missing_value_df = pd.DataFrame({'column_name': df.columns,
                                    'num_missing': df.isnull().sum(),
                                    'num_present': len(df) - df.isnull().sum(),
                                    'percent_missing': percent_missing})
    missing_value_df.sort_values('percent_missing', inplace=True)
    # missing_value_df = missing_value_df[missing_value_df.percent_missing != 0]
    return missing_value_df


def get_cols_missing_percentage(cutoff_percentage, df, name, show_missing=True):
    df_missing_value = get_missing_values_cols(df)
    df_missing_value.to_csv(f'./feature_selection/missing_percentage_{name}.csv', index=False)
    df_missing_value = df_missing_value[df_missing_value.percent_missing >= cutoff_percentage]
    if show_missing:
        print(
            f'{len(df_missing_value)} columns were over {cutoff_percentage} missing. This is the list of columns: {df_missing_value["column_name"].to_list()}')
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


def select_numeric_columns(df):
    numeric_columns = df.select_dtypes(include=['float64','int64']).columns.tolist()
    return numeric_columns


def convert_numeric_to_float16(df):
    numeric_cols = select_numeric_columns(df)
    df[numeric_cols] = df[numeric_cols].astype(np.float16)
    return df


def resample_max(df: pd.DataFrame, label: str, n_max_per_class, is_test=False, replace=True) -> pd.DataFrame:
    df_majority = df[df[label] ==0]
    df_minority = df[df[label] ==1]

    # downsample df_majority class
    df_majority_downsampled = resample(df_majority,
                                    replace=False,     # sample with replacement
                                    n_samples=n_max_per_class,    # to match majority class
                                    # random_state=44
                                    )

    df_minority_upsampled = df_minority
    if not is_test:
        # Upsample minority class
        df_minority_upsampled = resample(df_minority,
                                        replace=replace,     # sample with replacement
                                        n_samples=n_max_per_class,    # to match majority class
                                        # random_state=44
                                        )

    # Combine majority class with upsampled minority class
    df_sampled = pd.concat([df_majority_downsampled, df_minority_upsampled])

    return df_sampled


def resample_class(df: pd.DataFrame, label: str, label_val: object, n_max_per_class: int) -> pd.DataFrame:
    df_majority = df[df[label] == label_val]
    df_minority = df[df[label] != label_val]

    # downsample df_majority class
    df_majority = resample(df_majority,
                                    replace=False,     # sample with replacement
                                    n_samples=n_max_per_class,    # to match majority class
                                    # random_state=44
                                    )
    return pd.concat([df_majority, df_minority])