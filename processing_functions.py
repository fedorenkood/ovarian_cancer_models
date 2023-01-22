import numpy as np
import pandas as pd
# Visualization
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
import plotly.express as px
px_template = "simple_white"
# Split data
from sklearn.model_selection import train_test_split, cross_validate


from imblearn.over_sampling import SMOTE

from sklearn.preprocessing import StandardScaler 

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier

# Libtune to tune model, get different metric scores
from sklearn import datasets, linear_model, metrics
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score, ConfusionMatrixDisplay, PrecisionRecallDisplay, precision_score, recall_score, f1_score, classification_report, roc_curve, plot_roc_curve, auc, precision_recall_curve, plot_precision_recall_curve, average_precision_score
from sklearn.model_selection import cross_val_score
from collections import Counter
from sklearn.model_selection import GridSearchCV


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
    df_scaled = df.drop('Diabetes_binary', axis=1)
    df_scaled = sc.fit_transform(df_scaled)
    df_scaled = pd.DataFrame(df_scaled, columns=df.columns[1:])
    df_scaled.insert(0, 'Diabetes_binary', df['Diabetes_binary'].to_numpy())
    return df_scaled

def scaling_tranform(X_train, X_test):
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)
    return X_train, X_test

def run_classifier(classifier, X_train, X_test, y_train, y_test):
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)
    y_prob = classifier.predict_proba(X_test)[:,1]
    auc, accuracy = performance_analysis(y_pred, y_prob, y_test)
    return auc, accuracy

def performance_analysis(y_pred, y_prob, y_test):
    report = pd.DataFrame(classification_report(y_test, y_pred, output_dict=True)).transpose().iloc[0:2,:]
    print(report)
    auc = roc_auc_score(y_test, y_prob)
    accuracy = accuracy_score(y_test, y_pred)
    print(f'ROC AUC score: {auc}')
    print(f'Accuracy Score: {accuracy}')
    f, ax = plt.subplots(1, 3, figsize=(16, 5))
    plt.yticks(rotation = 0)
    display_confusion_matrix(ax[0], y_test, y_pred)

    display_auc(ax[1], y_test, y_prob)

    display_precision_recall(ax[2], y_prob, y_test)
    plt.show()
    return auc, accuracy


def get_accuracy(y_test, y_pred):
    return accuracy_score(y_test, y_pred)*100

def display_confusion_matrix(ax, y_test, y_pred):
    cm = pd.DataFrame(confusion_matrix(y_test, y_pred), columns=['Predicted Healthy', 'Predicted Diabetic'], index=['Healthy', 'Diabetic'])
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

