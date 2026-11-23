
# -*- coding: utf-8 -*-
"""
Brier Score and Calibration Curve
Binary Classification: GNN Model

Class 0: Non-inhibitor
Class 1: Inhibitor

Input columns:
Actual, Predicted, Prob

Prob = predicted probability of Class 1
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import brier_score_loss
from sklearn.calibration import calibration_curve

INPUT_FILE = "test_actual_vs_predicted.csv"
OUTPUT_DIR = "Brier_Calibration_Results"
N_BINS = 5
BINNING_STRATEGY = "quantile"
os.makedirs(OUTPUT_DIR, exist_ok=True)
df = pd.read_csv(INPUT_FILE)
required_cols = ["Actual", "Prob"]

df = df.dropna(subset=required_cols).copy()
y_true = df["Actual"].astype(int).to_numpy()
y_prob = df["Prob"].astype(float).to_numpy()

if len(y_true) == 0:
    raise ValueError("No valid samples found.")

if not np.isin(y_true, [0, 1]).all():
    raise ValueError("Actual must contain 0 and/or 1.")

if not np.isfinite(y_prob).all():
    raise ValueError("Prob contains non-finite values.")

if not np.all((y_prob >= 0) & (y_prob <= 1)):
    raise ValueError("Probabilities must be between 0 and 1.")

brier_score = brier_score_loss(y_true, y_prob)
prevalence = np.mean(y_true)
baseline_probs = np.full(len(y_true), prevalence)
brier_reference = brier_score_loss(y_true,baseline_probs)

if brier_reference > 0:
    brier_skill = 1 - (brier_score / brier_reference)
else:
    brier_skill = np.nan

print("\n========== BRIER SCORE RESULTS ==========")
print(f"Number of samples: {len(y_true)}")
print(f"Inhibitor prevalence: {prevalence:.4f}")
print(f"Brier Score: {brier_score:.6f}")
print(f"Reference Brier Score: {brier_reference:.6f}")
print(f"Brier Skill Score: {brier_skill:.6f}")

fraction_positive, mean_predicted = calibration_curve(y_true,y_prob,n_bins=N_BINS,strategy=BINNING_STRATEGY)
bin_edges = np.percentile(y_prob,np.linspace(0, 100, N_BINS + 1))
bin_ids = np.searchsorted(bin_edges[1:-1], y_prob,side="left")
counts = np.bincount(bin_ids,minlength=N_BINS)

calibration_data = pd.DataFrame({
    "Mean_Predicted_Probability": mean_predicted,
    "Observed_Fraction_Inhibitors": fraction_positive,
    "Samples_in_Bin": counts[counts > 0]})


results_df = pd.DataFrame([{
    "Dataset": "Independent Test",
    "N_Samples": len(y_true),
    "N_Bins": N_BINS,
    "Binning_Strategy": BINNING_STRATEGY,
    "Brier_Score": brier_score,
    "Positive_Class_Prevalence": prevalence,
    "Reference_Brier_Score": brier_reference,
    "Brier_Skill_Score": brier_skill}])

results_df.to_csv(os.path.join(OUTPUT_DIR, "Brier_Score_Results.csv"),index=False,float_format="%.6f")
calibration_data.to_csv(os.path.join(OUTPUT_DIR, "Calibration_Curve_Data.csv"),index=False,float_format="%.6f")
print(calibration_data.to_string(index=False))
