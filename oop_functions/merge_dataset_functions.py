import pandas as pd

from .util_functions import get_unique_combinations, remove_featues_startswith


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
        df['ovar_observe_year'] = base_year
        df_final = pd.concat([df_final, df])
        df_final = df_final.drop_duplicates()
    # Add a feature that says whether person was screened or not
    condition = df_final['plco_id'].isin(screen_df['plco_id'])
    df_final['was_screened'] = 0
    df_final.loc[condition, 'was_screened'] = 1
    return df_final