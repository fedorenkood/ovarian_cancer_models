from oop_functions.analytics_cv_util import *
from oop_functions.util_functions import *
from oop_functions.experiment_helper import *


def load_cv_analytics_util_see_stats(filesuffix):
    cv_analytics_util = CvAnalyticsUtil.load_cv_analytics_utils(filesuffix)
    cv_analytics_util.get_cv_report();
    cv_analytics_util.display_graph()
    return cv_analytics_util


def evaluate_threshold(threshold, predictions, targets):
    # Create binary predictions based on the threshold
    binary_predictions = (predictions >= threshold).astype(int)
    
    # Calculate evaluation metrics
    tp = np.sum((binary_predictions == 1) & (targets == 1))
    fp = np.sum((binary_predictions == 1) & (targets == 0))
    tn = np.sum((binary_predictions == 0) & (targets == 0))
    fn = np.sum((binary_predictions == 0) & (targets == 1))
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    true_positive_rate = tp / (tp + fn) if (tp + fn) > 0 else 0
    false_positive_rate = fp / (fp + tn) if (fp + tn) > 0 else 0
    
    return tp, fp, tn, fn, precision, recall, true_positive_rate, false_positive_rate


def get_per_thereshold_metrics(df, probability, label, thresholds = np.linspace(0, 1, 101)):
    # Initialize lists to store results for each threshold
    tp_list = []
    fp_list = []
    tn_list = []
    fn_list = []
    precision_list = []
    recall_list = []
    tpr_list = []
    fpr_list = []

    # Iterate through each threshold and calculate metrics
    for threshold in thresholds:
        tp, fp, tn, fn, precision, recall, tpr, fpr = evaluate_threshold(threshold, df[f'{label}_prob'], df[label])
        
        tp_list.append(tp)
        fp_list.append(fp)
        tn_list.append(tn)
        fn_list.append(fn)
        precision_list.append(precision)
        recall_list.append(recall)
        tpr_list.append(tpr)
        fpr_list.append(fpr)

    # Create a new DataFrame to store the results
    results_df = pd.DataFrame({
        'Threshold': thresholds,
        'True_Positive': tp_list,
        'False_Positive': fp_list,
        'True_Negative': tn_list,
        'False_Negative': fn_list,
        'Precision': precision_list,
        'Recall': recall_list,
        'True_Positive_Rate': tpr_list,
        'False_Positive_Rate': fpr_list
    })
    return results_df


def extract_full_dataset_filter_cancer(full_dataset: pd.DataFrame):
    full_dataset = full_dataset.copy()
    full_dataset = full_dataset[full_dataset['ovar_cancer'] == 1]
    full_dataset = full_dataset[full_dataset['ovar_cancer_years'] <= 5]
    full_dataset = full_dataset[full_dataset['study_yr'] != -1]
    return full_dataset


def extract_full_dataset_filter_no_cancer(full_dataset: pd.DataFrame):
    full_dataset = full_dataset.copy()
    full_dataset = full_dataset[full_dataset['ovar_cancer'] == 0]
    full_dataset = full_dataset[full_dataset['ovar_observe_year'] <= 6]
    full_dataset = full_dataset[full_dataset['study_yr'] != -1]
    return full_dataset


def plot_diff_in_confidence(full_dataset, title=''):
    
    grouped_df = full_dataset.groupby(['plco_id'])
    diff_df = []

    for key, item in grouped_df:
        diff = {}
        group = grouped_df.get_group(key)
        group = group.sort_values('study_yr').reset_index()
        diff['plco_id'] = group.loc[0, 'plco_id']
        years = group['study_yr'].unique()
        max_year = 6 - len(years)
        # max_year = 0
        for year in range(len(years) - 1):
            diff[f'cancer_in_next_1_years_{int(max_year + year+1)}-{int(max_year + year)}'] = group.loc[year+1, 'cancer_in_next_1_years_prob'] - group.loc[year, 'cancer_in_next_1_years_prob']
            # if len(years) == 5:
            #     print(diff)
        diff_df.append(diff)

    ordered_cols = []
    for i in range(5):
        ordered_cols.append(f'cancer_in_next_1_years_{i+1}-{i}')
        
    diff_df = pd.DataFrame(diff_df)

    diff_df = diff_df[['plco_id'] + ordered_cols]
    print_df(diff_df.describe().T)
    x = list(range(-1, -6, -1))
    stats = diff_df.describe()
    y = np.array(stats.loc['mean'].to_list())
    ci = np.array(stats.loc['std'].to_list())
    plt.plot(x, y)
    plt.fill_between(x, (y-ci), (y+ci), color='b', alpha=.1)
    plt.ylabel("Probability of getting cancer")
    plt.xlabel("Years before getting cancer")
    plt.ylim([0, 1])
    plt.title(title)


def plot_change_in_confidence(full_dataset, title = '', label = f'cancer_in_next_1_years'):    
    grouped_df = full_dataset.groupby(['plco_id'])
    diff_df = []
    ordered_cols = set()

    for key, item in grouped_df:
        diff = {}
        group = grouped_df.get_group(key)
        group = group.sort_values('ovar_observe_year').reset_index()
        diff['plco_id'] = group.loc[0, 'plco_id']
        years = group['ovar_observe_year'].unique()
        max_year = max(years)
        # print(min(years))
        # max_year = 0
        for i in range(len(years)):
            year = years[i]
            col = f'{label}_{int(year - max_year - 1)}'
            diff[col] = group.loc[i, f'{label}_prob']
            ordered_cols.add(col)
            # if len(years) == 5:
            #     print(diff)
        diff_df.append(diff)

    ordered_cols = list(ordered_cols)
    ordered_cols.sort(reverse=True)

    diff_df = pd.DataFrame(diff_df)

    diff_df = diff_df[['plco_id'] + ordered_cols]
    print_df(diff_df.describe().T)
    x = sorted(list(range(-1, -7, -1)))
    stats = diff_df.describe()
    y = np.array(stats.loc['mean'].to_list())
    ci = np.array(stats.loc['std'].to_list())
    plt.plot(x, y)
    plt.fill_between(x, (y-ci), (y+ci), color='b', alpha=.1)
    plt.ylabel("Probability of getting cancer")
    plt.xlabel("Years before getting cancer")
    plt.ylim([0, 1])
    plt.title(title)


def get_screened_first_5_no_process_dataset(label = f'cancer_in_next_1_years'):
    processed_data = pd.read_csv('./processed_dataset/recent_propagated_dataset.csv', index_col=0)
    processed_data = convert_numeric_to_float16(processed_data)
    experiment_data_helper: ExperimentDataHelper = ExperimentDataHelperScreenedFirst5(processed_data, label, ['cancer_'], train_size = 15000)
    return experiment_data_helper.source_df
