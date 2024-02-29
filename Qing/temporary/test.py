# import pickle
# import numpy as np
# from scipy.stats import entropy
# from tqdm import tqdm
# import os
# import json
# from sklearn.metrics import precision_recall_curve, auc
# import matplotlib.pyplot as plt
# import random
# import pandas as pd
# import csv
# import joblib
# from sklearn.metrics import roc_auc_score
#
#
# def unroll_pred(scores, indices):
#     unrolled = []
#     for idx in indices:
#         unrolled.extend(scores[idx])
#     return unrolled
#
# def get_PR_with_human_labels(preds, human_labels, pos_label=1, oneminus_pred=False):
#     # indices = [k for k in human_labels.keys()]
#     # unroll_preds = unroll_pred(preds, indices)
#     # if oneminus_pred:
#     #     unroll_preds = [1.0-x for x in unroll_preds]
#     # unroll_labels = unroll_pred(human_labels, indices)
#     # assert len(unroll_preds) == len(unroll_labels)
#     # print("len:", len(unroll_preds))
#     P, R, thre = precision_recall_curve(human_labels, preds, pos_label=pos_label)
#     # auroc = roc_auc_score(unroll_labels, unroll_preds)
#     return P, R
#
# def print_AUC(P, R):
#     print("AUC: {:.2f}".format(auc(R, P)*100))
#
#
# df = pd.read_csv('../result/m_hal/m_hal_df.csv')
# average_logprob_scores, average_entropy_scores, lowest_logprob_scores, highest_entropy_scores, human_label_detect_True = df['average_logprob'].tolist(), df['average_entropy'].tolist(), df['lowest_logprob'].tolist(), df['highest_entropy'].tolist(), df['labels'].tolist()
#
# human_label_detect_True_value = []
# for label in human_label_detect_True:
#     if label == 'ACCURATE' or label == 'ANALYSIS':
#         true_score = 1.0
#     elif label == 'INACCURATE':
#         true_score = 0.0
#     human_label_detect_True_value.append(true_score)
#
#
# def analysis_sentence_level_info(average_logprob_scores, average_entropy_scores, lowest_logprob_scores, highest_entropy_scores, human_label_detect_True):
#     # True
#     # uncertainty
#     Pb_average_logprob, Rb_average_logprob = get_PR_with_human_labels(average_logprob_scores,
#                                                                       human_label_detect_True_value, pos_label=1,
#                                                                       oneminus_pred=False)
#     Pb_average_entropy, Rb_average_entropy = get_PR_with_human_labels(average_entropy_scores,
#                                                                       human_label_detect_True_value, pos_label=1,
#                                                                       oneminus_pred=True)
#     Pb_lowest_logprob, Rb_lowest_logprob = get_PR_with_human_labels(lowest_logprob_scores, human_label_detect_True_value,
#                                                                     pos_label=1, oneminus_pred=False)
#     Pb_highest_entropy, Rb_highest_entropy = get_PR_with_human_labels(highest_entropy_scores,
#                                                                       human_label_detect_True_value, pos_label=1,
#                                                                       oneminus_pred=True)
#
#     print("-----------------------")
#     print("Baseline1: Avg(logP)")
#     print_AUC(Pb_average_logprob, Rb_average_logprob)
#     print("-----------------------")
#     print("Baseline2: Avg(H)")
#     print_AUC(Pb_average_entropy, Rb_average_entropy)
#     print("-----------------------")
#     print("Baseline3: Max(-logP)")
#     print_AUC(Pb_lowest_logprob, Rb_lowest_logprob)
#     print("-----------------------")
#     print("Baseline4: Max(H)")
#     print_AUC(Pb_highest_entropy, Rb_highest_entropy)
#
#     arr = []
#     for v in human_label_detect_True_value:
#         arr.append(v)
#     random_baseline = np.mean(arr)
#
#     # with human label, Detecting Non-factual*
#     plt.figure(figsize=(5.5, 4.5))
#     plt.hlines(y=random_baseline, xmin=0, xmax=1.0, color='grey', linestyles='dotted', label='Random')
#     plt.plot(Rb_average_logprob, Pb_average_logprob, '-', label='Avg(logP)')
#     plt.plot(Rb_average_entropy, Pb_average_entropy, '-', label='Avg(H)')
#     plt.plot(Rb_lowest_logprob, Pb_lowest_logprob, '-', label='Max(-logP)')
#     plt.plot(Rb_highest_entropy, Pb_highest_entropy, '-', label='Max(H)')
#     plt.legend()
#     plt.ylabel("Precision")
#     plt.xlabel("Recall")
#     plt.show()
#
#
# analysis_sentence_level_info(average_logprob_scores, average_entropy_scores, lowest_logprob_scores,
#                                  highest_entropy_scores, human_label_detect_True)

from sklearn.metrics import precision_score

# Example ground truth labels and predictions
true_labels = [True, False, True, True, False]
predictions = [True, False, False, True, True]

# Calculate precision
precision = precision_score(true_labels, predictions, pos_label=True)

print("Precision:", precision)
