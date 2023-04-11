import itertools

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedKFold, train_test_split

from oop_functions.util_functions import resample_max, remove_featues_startswith, select_numeric_columns

px_template = "simple_white"

# https://scikit-learn.org/stable/modules/classes.html#module-sklearn.feature_selection

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


def impute_data(X_train: pd.DataFrame, X_test: pd.DataFrame, strategy, label):
    # Fill ovarycyst variables with 0 if they are in screen dataset (e.g. were screened)
    fill_const = {
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
    const_val_cols = list(fill_const.keys())
    # print(X_train[list(fill_const.keys())].describe().T)
    # X_train.loc[X_train['was_screened'] == 1] = X_train.loc[X_train['was_screened'] == 1].fillna(fill_const)
    # X_test.loc[X_test['was_screened'] == 1] = X_test.loc[X_test['was_screened'] == 1].fillna(fill_const)
    X_train = X_train.fillna(fill_const)
    X_test = X_test.fillna(fill_const)
    try: 
        X_train[const_val_cols] = X_train[const_val_cols].astype(np.int16)
        X_test[const_val_cols]  = X_test[const_val_cols].astype(np.int16)
        # Impute particular values with means
        mean_imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
        mean_imputer.fit(X_train[const_val_cols])
        # print_df(X_train[const_val_cols].describe().T)
        # print(X_test[const_val_cols].shape)
        # print(mean_imputer.transform(X_test[const_val_cols]).shape)
        X_test[const_val_cols] = mean_imputer.transform(X_test[const_val_cols])
        X_train[const_val_cols] = mean_imputer.transform(X_train[const_val_cols])
    except:
        pass
    

    # Others with whatever strategy we decide
    mean_imputer = SimpleImputer(missing_values=np.nan, strategy=strategy)
    mean_imputer.fit(X_train)
    X_test = mean_imputer.transform(X_test)
    X_train = mean_imputer.transform(X_train)

    # labeled_impute = LabeledImmpute()
    # labeled_impute.fit(X_train, label)
    # X_train = labeled_impute.transform(X_train, label, strategy)
    return X_train, X_test


def split_xy_drop(df, label, drop_cols):
    y = df[label]
    X = df.drop([label, *drop_cols], axis=1)
    return X, y


def process_train_test_split(source_df, train, test, label, train_size, test_size, strategy, stats=True, differentiate_confusion_matrix_over=None):
    train = source_df[source_df['plco_id'].isin(train['plco_id'])]
    test = source_df[source_df['plco_id'].isin(test['plco_id'])]
    # train = train.merge(source_df, how='left')
    # test = test.merge(source_df, how='left')
    y_train = train[label]
    y_test = test[label]
    # Printing stats
    if stats:
        print(f'Distribution of labels based on duplicate plco_id: {np.sum(y_test)/(np.sum(y_train) + np.sum(y_test))}')
    # drop non-cancer records without screen records
    # condition = (train['was_screened'] == 1) | (train['ovar_cancer'] == 1)
    # train = train[condition]
    train = train.drop(['plco_id'], axis=1)
    test = test.drop(['plco_id'], axis=1)
    train = remove_featues_startswith(train, ['ovar_', 'cancer_'], [label, 'ovar_histtype', 'ovar_behavior'], show_removed=False)
    test = remove_featues_startswith(test, ['ovar_', 'cancer_'], [label, 'ovar_histtype', 'ovar_behavior'], show_removed=False)
    # Perform imputation before oversampling
    columns = train.columns
    train, test = impute_data(train, test, strategy, label)
    # get_cols_missing_percentage(80, train, 'train', show_missing=True)

    train = pd.DataFrame(train, columns=columns)
    test = pd.DataFrame(test, columns=columns)
    # Perform oversamping
    train = resample_max(train, label, train_size, replace=True)
    # test = resample_max(test, label, test_size, replace=True)

    drop_cols = ['was_screened', 'ovar_histtype', 'ovar_behavior']
    differentiated_test_sets = []
    if differentiate_confusion_matrix_over:
        filtered_on = list(itertools.chain.from_iterable([zip([key]*len(vals), vals) for key, vals in differentiate_confusion_matrix_over.items()]))
        for col, values in filtered_on:
            filtered_test = test[test[col].isin(values)]
            X_test_filtered, y_test_filtered = split_xy_drop(filtered_test, label, drop_cols)
            differentiated_test_sets.append((X_test_filtered, y_test_filtered))


    X_train, y_train = split_xy_drop(train, label, drop_cols)
    X_test, y_test = split_xy_drop(test, label, drop_cols)

    if differentiate_confusion_matrix_over:
        return X_train, X_test, y_train, y_test, differentiated_test_sets
    return X_train, X_test, y_train, y_test


def process_and_impute_for_label_kfold(source_df, label, strategy, n_max_per_class=10000, num_folds=10, stats=True, differentiate_confusion_matrix_over=None):
    train_size = int(n_max_per_class * 0.8)
    test_size  = int(n_max_per_class * 0.2)
    train_fold_size  = int((num_folds-1) * train_size / num_folds)
    test_fold_size  = int(train_size / num_folds)

    # remove features starting with cancer so that we could drop labels that are nan (e.g. people get cancer later on)
    source_df = remove_featues_startswith(source_df, ['cancer_'], [label], show_removed=False)
    source_df = source_df[source_df[label].notnull()]
    print(len(source_df))
    
    # One person should not appear in train and test data since there are duplicates of a person
    # we splits of data on person id and then oversample from that sample 
    # this line of code determines whether the model is leaking info or not
    unique_id_df = source_df[['plco_id', label]].drop_duplicates(subset='plco_id')
    X_train_unique, X_test_unique, y_train, y_test = train_test_split(unique_id_df, unique_id_df[label], test_size = 0.2)

    # Printing stats
    # print_records_vs_unique(X_train_unique, 'plco_id', f'Train set', print_vals=True)
    # print_records_vs_unique(X_test_unique, 'plco_id', f'Test set', print_vals=True)
    print(f'Distribution of labels based on unique plco_id: {np.sum(y_test)/(np.sum(y_train) + np.sum(y_test))}')
    
    train_test_lambda = lambda: process_train_test_split(source_df, X_train_unique, X_test_unique, label, train_size, test_size, strategy, stats=stats, differentiate_confusion_matrix_over=differentiate_confusion_matrix_over)
    # create list of lambdas for each fold
    # Cross validation: https://vitalflux.com/k-fold-cross-validation-python-example/
    strtfdKFold = StratifiedKFold(n_splits=num_folds)
    kfold = strtfdKFold.split(unique_id_df, unique_id_df[label])
    k_fold_lambdas = []
    for k, (train, test) in enumerate(kfold):
        train = unique_id_df.iloc[train, :]
        test = unique_id_df.iloc[test, :]
        # print_records_vs_unique(train, 'plco_id', f'Train cv fold {k}', print_vals=True)
        # print_records_vs_unique(test, 'plco_id', f'Test cv fold {k}', print_vals=True)
        k_fold_lambdas.append(lambda: process_train_test_split(source_df, train, test, label, train_fold_size, test_fold_size, strategy, stats=False))

    return train_test_lambda, k_fold_lambdas


