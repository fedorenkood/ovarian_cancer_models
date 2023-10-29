from __future__ import annotations

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.metrics import auc, \
    PrecisionRecallDisplay

from oop_functions.util_functions import get_roc_threshold_point


class VisualizationUtil:
    def __init__(self) -> None:
        pass

    def display_confusion_matrix(self, ax: plt.axis, cm: pd.DataFrame) -> VisualizationUtil:
        sns.heatmap(cm, annot = True, fmt = 'd', cbar = False, ax=ax)
        # ax.set_title('Confusion Matrix')
        return self

    def display_roc_graph(self, ax: plt.axis, fpr: np.array, tpr: np.array, thresholds: np.array, tpr_std: np.array = None, label='', color=None, linestyle='solid', marker=None, roc_auc=None) -> VisualizationUtil:
        if roc_auc is None:
            roc_auc = auc(fpr, tpr)
        if len(label) > 0:
            label += '. '
        label = label + 'AUC = %0.3f' % roc_auc
        if color:
            ax.plot(fpr, tpr, label = label, color=color)
        else:
            ax.plot(fpr, tpr, label = label, linestyle=linestyle, marker=marker)
        ax.plot([0, 1], [0, 1], linestyle = '-.', color = 'gray')
        if tpr_std is not None:
            tpr_upper = np.clip(tpr+tpr_std, 0, 1)
            tpr_lower = np.clip(tpr-tpr_std, 0, 1)
            ax.fill_between(fpr, tpr_lower, tpr_upper, color='b', alpha=.1, label='Confidence Interval')

        # Put a legend to the right of the current axis
        ax.legend(loc='center left', bbox_to_anchor=(0.2, 0.2))

        ax.set_ylabel('TP Rate')
        ax.set_xlabel('FP Rate')
        # ax.set_title('ROC AUC Curve')
        return self

    def display_roc_threshold(self, ax: plt.axis, fpr: np.array, tpr: np.array, thresholds: np.array, tpr_std: np.array = None) -> VisualizationUtil:
        optimal_threshold, fpr_optimal, tpr_optimal = get_roc_threshold_point(fpr, tpr, thresholds)
        ax.plot(fpr_optimal, tpr_optimal, 'ro', label=f'Optimal threshold: {round(optimal_threshold, 2)}')
        ax.legend()
        return self

    def display_precision_recall(self, ax: plt.axis, precision: np.array, recall: np.array, std: np.array = None, label='', color=None, linestyle='solid', marker=None) -> VisualizationUtil:
        disp = PrecisionRecallDisplay(precision=precision, recall=recall)
        disp.plot(ax=ax, label=label, linestyle=linestyle, marker=marker)
        if std is not None:
            precision_upper = np.clip(precision+std, 0, 1)
            precision_lower = np.clip(precision-std, 0, 1)
            ax.fill_between(recall, precision_lower, precision_upper, color='b', alpha=.1, label='Confidence Interval')

        if len(label) > 0:
            # Put a legend to the right of the current axis
            ax.legend(loc='center left', bbox_to_anchor=(0.1, 0.8))
        
        ax.set_ylabel('Precision')
        ax.set_xlabel('Recall')
        # ax.set_title('Precision-Recall Curve')
        return self
