import pandas as pd
from matplotlib import pyplot as plt

from .util_functions import get_cols_missing_percentage, print_df, get_unique_combinations


def impute_with_val(df, columns, val):
    for col in columns:
        df[col] = df[col].fillna(val)
    return df


def print_records_vs_unique(df, col, dataset_name, print_vals=True):
    # Get unique IDs
    if print_vals:
        print(f"Num of records in {dataset_name} dataset: {len(df)}")
        print(f"Num of unique {col} in {dataset_name} dataset: {len(df[col].unique())}")
    return len(df), len(df[col].unique())


def print_records_vs_unique_for(df, col, dataset_name, on):
    print_records_vs_unique(df, col, dataset_name)
    # Look at unique IDs for each of the years of study for screen dataset
    for val in sorted(df[on].unique()):
        # Get unique IDs
        print_records_vs_unique(df[df[on] == val], col, f'{dataset_name}.{on}={val}')


# Prints the table of missing values with columns for df filtered for each of the values of on_col
def df_missing_val_distribution_over_col(df, on_col, cutoff_percentage, title, show_missing=True):
    iterate_over_on_col = sorted(df[on_col].unique())
    suffixes = []
    # since get_cols_missing_percentage will produce columns with the same names, we want to create suffixes to differentiate over on_col values
    for on_col_val in iterate_over_on_col:
        suffixes.append(f'_{on_col}_{on_col_val}')
    # We are going to merge all of the
    df_missing = None
    for on_col_val in iterate_over_on_col:

        df_on_col = df[df[on_col] == on_col_val]

        df_missing_on_col = get_cols_missing_percentage(cutoff_percentage, df_on_col,
                                                        f'{title} with {on_col}_{on_col_val}',
                                                        show_missing=show_missing)

        if df_missing is not None:
            df_missing = df_missing.merge(df_missing_on_col, on='column_name', how='inner', suffixes=suffixes)
        else:
            df_missing = df_missing_on_col
    print_df(df_missing)


def df_filter_val_distribution(sourse_df, on_col, make_unique_over_cols, name, hist=True, cutoff_percentage=0,
                               join='outer', suffixes=['_cancer', '_no_cancer']):
    unique_vals_list = [sorted(sourse_df[make_unique_over_col].unique()) for make_unique_over_col in
                        make_unique_over_cols]
    unique_combinations_col_vals = get_unique_combinations(unique_vals_list)
    merged_df = None
    num_records, num_unique = print_records_vs_unique(sourse_df, 'plco_id', name)
    unique_combinations_col_vals = sorted(unique_combinations_col_vals)
    unique_records = [{} for _ in range(len(unique_combinations_col_vals) + 1)]
    unique_records[0] = {
        'filtered_on': f'{name} total'
    }

    iterate_over_on_col = sorted(sourse_df[on_col].unique())
    for on_col_val in iterate_over_on_col:
        num_records, num_unique = print_records_vs_unique(sourse_df[sourse_df[on_col] == on_col_val], 'plco_id',
                                                          f'{name} with {on_col}_{on_col_val}')
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
        unique_records[j + 1]['filtered_on'] = title
        for on_col_val in iterate_over_on_col:
            df_on_col = df[df[on_col] == on_col_val]
            num_records, num_unique = print_records_vs_unique(df_on_col, 'plco_id',
                                                              f'{title} with {on_col}_{on_col_val}')
            df_missing_on_col = get_cols_missing_percentage(cutoff_percentage, df_on_col,
                                                            f'{title} with {on_col}_{on_col_val}')
            unique_records[j + 1][f'num_records_{on_col}:{on_col_val}'] = num_records
            unique_records[j + 1][f'num_unique_id_{on_col}:{on_col_val}'] = num_unique
            if hist:
                fig, ax = plt.subplots(1, 1, figsize=(20, 15))
                df.hist(ax=ax, bins=30)
                plt.show()
        df_missing_val_distribution_over_col(df, on_col, cutoff_percentage, title, show_missing=False)
        df = impute_with_val(df, df.columns, -1)
        print('\n\n')
    print_df(pd.DataFrame(unique_records).sort_values('filtered_on'))


def df_filter_val_distribution_on_cancer(source_df, make_unique_over_cols, name, personal_data_cancer,
                                         personal_data_no_cancer, hist=False, cutoff_percentage=0):
    df_cancer = source_df[source_df['plco_id'].isin(personal_data_cancer['plco_id'])]
    df_cancer['cancer'] = 1
    df_no_cancer = source_df[source_df['plco_id'].isin(personal_data_no_cancer['plco_id'])]
    df_no_cancer['cancer'] = 0
    df = pd.concat([df_cancer, df_no_cancer], axis=0)
    df_filter_val_distribution(df, 'cancer', make_unique_over_cols, name, hist=hist,
                               cutoff_percentage=cutoff_percentage)