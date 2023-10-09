from oop_functions.analytics_cv_util import *
from oop_functions.util_functions import *
from oop_functions.experiment_helper import *

from typing import List, Dict


def load_cv_analytics_util_see_stats(filesuffix):
    cv_analytics_util = CvAnalyticsUtil.load_cv_analytics_utils(filesuffix)
    cv_analytics_util.get_cv_report();
    cv_analytics_util.display_graph()
    return cv_analytics_util


def evaluate_threshold(threshold, predictions, targets):
    # Create binary predictions based on the threshold
    binary_predictions = (predictions > threshold).astype(int)
    
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
    grouped_df = full_dataset.groupby('plco_id')
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


def find_boundaries(arr, n):
    # Sort the array
    sorted_arr = np.sort(arr)
    
    # Calculate the percentiles
    percentiles = np.linspace(0, 100, n + 1)
    boundaries = np.percentile(sorted_arr, percentiles)
    
    # Remove duplicate boundaries
    return np.unique(boundaries).tolist()


def get_predefined_boundaries(full_dataset: pd.DataFrame, label, threshold):
    # full_dataset = full_dataset.copy()
    full_dataset = full_dataset[full_dataset[label] == 1]
    predictions = full_dataset[f'{label}_prob'].to_numpy()
    # sorted_arr = np.sort(predictions)
    # percentile_25 = np.percentile(sorted_arr, 25)
    # percentile_50 = np.percentile(sorted_arr, 50)
    # percentile_75 = np.percentile(sorted_arr, 75)
    # percentile_99 = np.percentile(sorted_arr, 99)
    # predictions = predictions[predictions > np.percentile(sorted_arr, 99.5)]
    boundaries = find_boundaries(predictions, 5)
    # boundaries = [
    #     0, 
    #     threshold, 
    #     np.percentile(sorted_arr, 99),
    #     np.percentile(sorted_arr, 99.25),
    #     np.percentile(sorted_arr, 99.5),
    #     np.percentile(sorted_arr, 99.6),
    #     np.percentile(sorted_arr, 99.7),
    #     np.percentile(sorted_arr, 99.8),
    #     np.percentile(sorted_arr, 99.9),
    #     1
    #     # *boundaries
    #     # np.percentile(sorted_arr, 99.5),
    #     # np.percentile(sorted_arr, 99.75),
    #     # np.percentile(sorted_arr, 99.9),
    #     # 1,
    #     ]
    boundaries[0] = -0.0001
    # boundaries.insert(0, 0)
    boundaries[-1] = 1
    # del boundaries[-1]
    print(boundaries)
    return boundaries


def bucket_predictions_by_thresholds(cv_analytics_util: CvAnalyticsUtil):
    threshold = cv_analytics_util.get_optimal_operating_point()
    label = cv_analytics_util.get_label()
    full_dataset = cv_analytics_util.get_dataset_with_predictions()
    boundaries = get_predefined_boundaries(full_dataset, label, threshold)
    per_thereshold_metrics = get_per_thereshold_metrics(full_dataset, f'{label}_prob', label, thresholds = boundaries)
    per_thereshold_metrics['bucket_positives'] = per_thereshold_metrics['True_Positive'].diff().abs()
    per_thereshold_metrics['bucket_negatives'] = per_thereshold_metrics['False_Positive'].diff().abs()
    per_thereshold_metrics['per_bucket_probability'] = per_thereshold_metrics['bucket_positives'] / (per_thereshold_metrics['bucket_positives'] + per_thereshold_metrics['bucket_negatives'])
    return per_thereshold_metrics


def plot_threhold_probabilities(per_thereshold_metrics):
    title = ""
    x = per_thereshold_metrics['Threshold'][1:]
    y = per_thereshold_metrics['per_bucket_probability'][1:]
    plt.plot(x, y, label="Probability of getting cancer")
    y = per_thereshold_metrics['Precision'][:-1]
    plt.plot(x, y, label="Precision")
    plt.ylabel("Probability of getting cancer")
    plt.xlabel("Threshold")
    plt.title(title)
    plt.ylim([0, 1])
    plt.legend()
    plt.show()


def create_buckets(thresholds):
    # Ensure the thresholds are sorted
    sorted_thresholds = sorted(thresholds)
    
    # Create buckets using the sorted thresholds
    buckets = [(sorted_thresholds[i], sorted_thresholds[i + 1]) for i in range(len(sorted_thresholds) - 1)]
    
    return buckets


def map_label_prob_to_bucket(per_thereshold_metrics, row, label):
    thresholds = per_thereshold_metrics['Threshold'].to_list()
    buckets = create_buckets(thresholds)
    for index, (left, right) in enumerate(buckets):
        if index == 0:
            left = -0.1
        if left < row[f'{label}_prob'] <= right:
            return per_thereshold_metrics.iloc[index + 1]['per_bucket_probability']
    return None  # Handle cases where label_prob is above all thresholds


def load_cv_analytics_utils(filesuffixes: List[str]) -> Dict[str, CvAnalyticsUtil]:
    cv_analytics_utils: Dict[str, CvAnalyticsUtil] = {}
    for filesuffix in filesuffixes:
        print(filesuffix)
        cv_analytics_util = CvAnalyticsUtil.load_cv_analytics_utils(filesuffix)
        cv_analytics_utils[filesuffix] = cv_analytics_util
        label = cv_analytics_util.get_label()
        if 'single' in filesuffix:
            cv_analytics_util.merge_in_dataset(get_screened_first_5_no_process_dataset(label = label))
    return cv_analytics_utils


def find_intersection(list_of_arrays):
    if not list_of_arrays:
        return None

    # Start with the first array as the initial intersection
    intersection = list_of_arrays[0]

    # Iterate through the remaining arrays and find their intersection
    for arr in list_of_arrays[1:]:
        intersection = np.intersect1d(intersection, arr)

    return intersection


def get_intersecting_indexes(cv_analytics_utils: Dict[str, CvAnalyticsUtil]) -> np.array:
    list_of_arrays = []
    for filesuffix, cv_analytics_util in cv_analytics_utils.items():
        full_dataset = cv_analytics_util.get_dataset_with_predictions()
        print(filesuffix)
        print(f'Number of records: {len(full_dataset)}')
        list_of_arrays.append(full_dataset['index'].astype(int).to_numpy())
    intersecting_indexes = find_intersection(list_of_arrays)
    print(f'Number of indersecting indexes: {len(intersecting_indexes)}')
    return intersecting_indexes


def commonized_datasets(filesuffixes) -> Dict[str, CvAnalyticsUtil]:
    cv_analytics_utils = load_cv_analytics_utils(filesuffixes)
    labels = [cv_analytics_util.get_label() for filesuffix, cv_analytics_util in cv_analytics_utils.items()]
    labels = set(labels)
    for label in labels:
        intersecting_indexes = get_intersecting_indexes({key: cv_analytics_util for key, cv_analytics_util in cv_analytics_utils.items() if cv_analytics_util.get_label() == label})
        for filesuffix, cv_analytics_util in cv_analytics_utils.items():
            if cv_analytics_util.get_label() == label:
                cv_analytics_util.keep_indexes(intersecting_indexes)
    return cv_analytics_utils
