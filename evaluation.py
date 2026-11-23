# -*- coding: utf-8 -*-
"""
Created on Thu Nov 19 10:55:41 2026

@author: anjut
"""


import os
import numpy as np
import pandas as pd

from sklearn.metrics import (accuracy_score,precision_recall_fscore_support,confusion_matrix,roc_auc_score,average_precision_score)

from scipy.stats import binomtest
files = "test_actual_vs_predicted.csv"
output_dir = "classification_results"
os.makedirs(output_dir, exist_ok=True)
confusion_results = []
overall_results = []
classwise_results = []

for dataset_name, file_path in files.items():
    df = pd.read_csv(file_path)
    required_columns = ["Actual", "Predicted", "Prob"]
    missing = [
        col for col in required_columns
        if col not in df.columns
    ]
    df = df.dropna(subset=required_columns).copy()
    y_true = df["Actual"].astype(int).to_numpy()
    y_pred = df["Predicted"].astype(int).to_numpy()
    y_prob = df["Prob"].astype(float).to_numpy()
    if not np.isin(y_true, [0, 1]).all():
        raise ValueError(f"Invalid Actual labels in {file_path}")
    if not np.isin(y_pred, [0, 1]).all():
        raise ValueError(f"Invalid Predicted labels in {file_path}")
    if len(y_true) == 0:
        raise ValueError(f"No valid rows in {file_path}")
    if not np.isfinite(y_prob).all() or not (
        (y_prob >= 0) & (y_prob <= 1)
    ).all():
        raise ValueError(f"Prob must be between 0 and 1 in {file_path}")

    tn, fp, fn, tp = confusion_matrix(y_true,y_pred,labels=[0, 1]).ravel()
    confusion_results.append({"Dataset": dataset_name,"TP": tp,"TN": tn,"FP": fp,"FN": fn})

    accuracy = accuracy_score(y_true, y_pred)
    weighted_precision, weighted_recall, weighted_f1, _ = (
        precision_recall_fscore_support(y_true,y_pred,labels=[0, 1],average="weighted",zero_division=0))
    n_samples = len(y_true)
    n_correct = int(np.sum(y_true == y_pred))

    ci = binomtest(k=n_correct,n=n_samples).proportion_ci(confidence_level=0.95,method="exact")
    ci_lower = ci.low
    ci_upper = ci.high
    ci_formatted = f"{ci_lower:.4f}–{ci_upper:.4f}"

    if len(np.unique(y_true)) == 2:
        auroc = roc_auc_score(y_true, y_prob)
        pr_auc = average_precision_score(y_true, y_prob)
    else:
        auroc = np.nan
        pr_auc = np.nan

    overall_results.append({
        "Dataset": dataset_name,
        "Acc": accuracy,
        "Weighted-Pr": weighted_precision,
        "Weighted-Re": weighted_recall,
        "Weighted-F1": weighted_f1,
        "95% CI": ci_formatted,
        "AU-ROC": auroc,
        "PR-AUC": pr_auc
    })

    precision, recall, f1, support = (precision_recall_fscore_support(y_true,y_pred,labels=[0, 1],average=None,zero_division=0))

    for i, class_label in enumerate([0, 1]):
        classwise_results.append({"Dataset": dataset_name,"Class": class_label,"Precision": precision[i],"Recall": recall[i],"F1-score": f1[i]})

confusion_df = pd.DataFrame(confusion_results)
overall_df = pd.DataFrame(overall_results)
classwise_df = pd.DataFrame(classwise_results)

confusion_file = os.path.join(output_dir,"confusion_matrix_results.csv")
overall_file = os.path.join(output_dir,"table1_overall_metrics.csv")
classwise_file = os.path.join(output_dir,"table2_classwise_metrics.csv")
confusion_df.to_csv(confusion_file, index=False)
overall_df.to_csv(overall_file,index=False,float_format="%.4f")
classwise_df.to_csv(classwise_file,index=False,float_format="%.4f")