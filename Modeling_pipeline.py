"""
Baseline (logit) + XGBoost modeling pipeline for the DPES 2023 late-decider
thesis. a stratified 80/20 train/test split held out before
any fitting, a logit baseline replicating Proulx et al. (2025)'s control
set + 2 extra variables, an XGBoost model tuned with 5-fold CV on the
training set only, and SHAP interpretation focused on cynicism and its
interaction with age.
"""
import numpy as np
import pandas as pd
import statsmodels.api as sm
import xgboost as xgb
import shap
import matplotlib.pyplot as plt
import pickle
from sklearn.model_selection import (
    train_test_split, StratifiedKFold, RandomizedSearchCV, cross_val_predict,
)
from sklearn.metrics import (
    f1_score, roc_auc_score, classification_report,
    average_precision_score, brier_score_loss, confusion_matrix,
    ConfusionMatrixDisplay,
)
from sklearn.calibration import calibration_curve

from preprocessing import load_and_prepare, CYNICISM_COL, TARGET_COL, CONTROL_COLS

RANDOM_STATE = 42

# ==============================================================================
# LOAD + BUILD FEATURES
# ==============================================================================
df_clean, norm_cols = load_and_prepare("DPES2023DATASET.sav")

# Only respondents with a defined target go into modeling (the "don't know
# yet" respondents are excluded)
model_df = df_clean.dropna(subset=[TARGET_COL]).copy()
print(f"Respondents with a valid target_shifter: {len(model_df)}")
print(f"Class balance:\n{model_df[TARGET_COL].value_counts(normalize=True).round(3)}")

# ==============================================================================
# TRAIN / TEST SPLIT
# Held out before ANY model fitting or tuning. Stratified so both sides
# keep the overall shifter rate. Touched exactly once, at final evaluation.
# ==============================================================================
train_df, test_df = train_test_split(
    model_df, test_size=0.2, stratify=model_df[TARGET_COL], random_state=RANDOM_STATE
)
print(f"\nTrain: {len(train_df)}  Test: {len(test_df)}")
print(f"Train shift rate: {train_df[TARGET_COL].mean():.3f}  "
      f"Test shift rate: {test_df[TARGET_COL].mean():.3f}")

# ==============================================================================
# BASELINE: LOGIT
# Proulx et al. (2025) used pooled logistic regression with a 7-band
# categorical age variable and their original 7 controls. This thesis adds
# two further  controls -- social media use (Q02f) and 
# leader-vs-
# program importance (P55) 
#
# statsmodels requires complete rows (no native NaN handling, unlike
# XGBoost below), so the baseline necessarily runs on a smaller,
# complete-case subset of train/test
# ==============================================================================
baseline_feature_cols = [CYNICISM_COL, "age_band"] + CONTROL_COLS


def build_logit_design(df, reference_columns=None):
    X = df[[CYNICISM_COL, "political_interest_reversed", "urbanization",
            "is_woman", "edu_class", "income_class",
            "social_media_use", "leader_program_importance"]].copy()
    X = pd.concat([
        X,
        pd.get_dummies(df["age_band"], prefix="age", drop_first=True),
        pd.get_dummies(df["marital_status"], prefix="marital", drop_first=True),
        pd.get_dummies(df["religion"], prefix="religion", drop_first=True),
    ], axis=1)
    X = sm.add_constant(X, has_constant="add")
    X = X.astype(float)
    if reference_columns is not None:
        X = X.reindex(columns=reference_columns, fill_value=0.0)
    return X


baseline_train = train_df.dropna(subset=baseline_feature_cols + [TARGET_COL])
baseline_test = test_df.dropna(subset=baseline_feature_cols + [TARGET_COL])
print(f"\nBaseline logit complete-case N: train={len(baseline_train)}, "
      f"test={len(baseline_test)}")

X_train_logit = build_logit_design(baseline_train)
X_test_logit = build_logit_design(baseline_test, reference_columns=X_train_logit.columns)
y_train_logit = baseline_train[TARGET_COL]
y_test_logit = baseline_test[TARGET_COL]

# Class weighting, matched to XGBoost's scale_pos_weight treatment below so
# the two models are compared on equal footing. 
#
# statsmodels' Logit has no built-in sample-weight argument, so this uses
# GLM with a Binomial family and freq_weights instead. freq_weights is
# designed for integer replication counts, not arbitrary weights, so
# standard errors / p-values here should be read as approximate
logit_scale_pos_weight = (y_train_logit == 0).sum() / (y_train_logit == 1).sum()
print(f"\nBaseline logit scale_pos_weight: {logit_scale_pos_weight:.2f}")
train_weights = np.where(y_train_logit == 1, logit_scale_pos_weight, 1.0)

logit_model = sm.GLM(
    y_train_logit, X_train_logit,
    family=sm.families.Binomial(), freq_weights=train_weights,
).fit()
print(logit_model.summary())

logit_pred_proba = logit_model.predict(X_test_logit)
logit_pred = (logit_pred_proba >= 0.5).astype(int)
logit_f1 = f1_score(y_test_logit, logit_pred)
logit_auc = roc_auc_score(y_test_logit, logit_pred_proba)
print(f"\nBaseline logit (test set) -- F1: {logit_f1:.3f}  ROC-AUC: {logit_auc:.3f}")

# ==============================================================================
# XGBOOST
# Nominal controls one-hot encoded; ordinal controls (edu_class,
# income_class, social_media_use, leader_program_importance) kept as
# single integer columns; missing values left as NaN for XGBoost's native
# handling
# ==============================================================================


def build_xgb_design(df, reference_columns=None):
    X = df[[CYNICISM_COL, "age", "political_interest_reversed", "urbanization",
            "is_woman", "edu_class", "income_class",
            "social_media_use", "leader_program_importance"]].copy()
    X = pd.concat([
        X,
        pd.get_dummies(df["marital_status"], prefix="marital"),
        pd.get_dummies(df["religion"], prefix="religion"),
    ], axis=1)
    if reference_columns is not None:
        X = X.reindex(columns=reference_columns, fill_value=0)
    return X


X_train = build_xgb_design(train_df)
X_test = build_xgb_design(test_df, reference_columns=X_train.columns)
y_train = train_df[TARGET_COL]
y_test = test_df[TARGET_COL]

# Class weighting for imbalance (shift rate ~29%).
scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
print(f"\nscale_pos_weight (class imbalance correction): {scale_pos_weight:.2f}")

# Hyperparameter search space -- a reasonable starting grid; narrow or
# widen based on what RandomizedSearchCV's best_params_ lands near.
param_dist = {
    "max_depth": [3, 4, 5, 6],
    "learning_rate": [0.01, 0.03, 0.05, 0.1],
    "n_estimators": [100, 200, 300, 500],
    "subsample": [0.7, 0.8, 0.9, 1.0],
    "colsample_bytree": [0.7, 0.8, 0.9, 1.0],
    "min_child_weight": [1, 3, 5],
}

xgb_clf = xgb.XGBClassifier(
    objective="binary:logistic",
    scale_pos_weight=scale_pos_weight,
    eval_metric="logloss",
    random_state=RANDOM_STATE,
    missing=np.nan,
)

# 5-fold CV for tuning, within the training set only -- the test set is
# never touched until final evaluation below.
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
search = RandomizedSearchCV(
    xgb_clf, param_distributions=param_dist, n_iter=40, scoring="f1",
    cv=cv, random_state=RANDOM_STATE, n_jobs=-1, verbose=1,
)
search.fit(X_train, y_train)

print(f"\nBest CV F1 (training folds only): {search.best_score_:.3f}")
print(f"Best params: {search.best_params_}")

best_model = search.best_estimator_
xgb_pred_proba = best_model.predict_proba(X_test)[:, 1]
xgb_pred = best_model.predict(X_test)
xgb_f1 = f1_score(y_test, xgb_pred)
xgb_auc = roc_auc_score(y_test, xgb_pred_proba)
print(f"\nXGBoost (test set) -- F1: {xgb_f1:.3f}  ROC-AUC: {xgb_auc:.3f}")
print("\nXGBoost classification report (test set):")
print(classification_report(y_test, xgb_pred))

# --- Overfitting check ---
# Compares training-set performance against the test-set numbers above.
# A training score dramatically higher than the test score would signal
# the model memorized the training data rather than generalizing.
xgb_train_pred = best_model.predict(X_train)
xgb_train_pred_proba = best_model.predict_proba(X_train)[:, 1]
xgb_train_f1 = f1_score(y_train, xgb_train_pred)
xgb_train_auc = roc_auc_score(y_train, xgb_train_pred_proba)
print(f"\nOverfitting check -- Train F1: {xgb_train_f1:.3f} (test: {xgb_f1:.3f})  "
      f"Train ROC-AUC: {xgb_train_auc:.3f} (test: {xgb_auc:.3f})")

# --- Save the final model ---

with open("xgb_model.pkl", "wb") as f:
    pickle.dump(best_model, f)
print("Saved xgb_model.pkl")

# ==============================================================================
# EVALUATION: baseline vs. XGBoost, side by side

# ==============================================================================

# --- Precision-Recall AUC ---
# ROC-AUC can look reasonable under class imbalance even when the model
# struggles specifically on the minority class (shifters). PR-AUC/average
# precision focuses on that
logit_pr_auc = average_precision_score(y_test_logit, logit_pred_proba)
xgb_pr_auc = average_precision_score(y_test, xgb_pred_proba)

# --- Calibration ---
# brier score
logit_brier = brier_score_loss(y_test_logit, logit_pred_proba)
xgb_brier = brier_score_loss(y_test, xgb_pred_proba)

print("\n=== Baseline vs. XGBoost (test set) ===")
print(f"{'Model':<12} {'N (test)':>10} {'F1':>8} {'ROC-AUC':>10} {'PR-AUC':>9} {'Brier':>8}")
print(f"{'Logit':<12} {len(baseline_test):>10} {logit_f1:>8.3f} {logit_auc:>10.3f} "
      f"{logit_pr_auc:>9.3f} {logit_brier:>8.3f}")
print(f"{'XGBoost':<12} {len(test_df):>10} {xgb_f1:>8.3f} {xgb_auc:>10.3f} "
      f"{xgb_pr_auc:>9.3f} {xgb_brier:>8.3f}")

# --- Fair same-sample comparison ---
# The table above compares XGBoost on its full test set against the
# logit's smaller complete-case test set ->not a paired comparison. This
# re-evaluates the SAME trained XGBoost model restricted to exactly the
# same respondents the baseline was tested on.
shared_test_idx = test_df.index.intersection(baseline_test.index)
X_test_shared = X_test.loc[test_df.index.isin(shared_test_idx)]
y_test_shared = y_test.loc[test_df.index.isin(shared_test_idx)]
xgb_shared_pred = best_model.predict(X_test_shared)
xgb_shared_pred_proba = best_model.predict_proba(X_test_shared)[:, 1]
xgb_shared_f1 = f1_score(y_test_shared, xgb_shared_pred)
xgb_shared_auc = roc_auc_score(y_test_shared, xgb_shared_pred_proba)
xgb_shared_pr_auc = average_precision_score(y_test_shared, xgb_shared_pred_proba)
print(f"\n=== Same-sample comparison (N={len(shared_test_idx)}, "
      f"baseline's complete-case respondents only) ===")
print(f"{'Logit':<12} {logit_f1:>8.3f} {logit_auc:>10.3f} {logit_pr_auc:>9.3f}")
print(f"{'XGBoost':<12} {xgb_shared_f1:>8.3f} {xgb_shared_auc:>10.3f} {xgb_shared_pr_auc:>9.3f}")

# --- Calibration curve plot ---
fig, ax = plt.subplots(figsize=(6, 6))
for name, y_true, y_proba in [
    ("Logit", y_test_logit, logit_pred_proba),
    ("XGBoost", y_test, xgb_pred_proba),
]:
    frac_pos, mean_pred = calibration_curve(y_true, y_proba, n_bins=10)
    ax.plot(mean_pred, frac_pos, marker="o", label=name)
ax.plot([0, 1], [0, 1], linestyle="--", color="grey", label="Perfectly calibrated")
ax.set_xlabel("Mean predicted probability")
ax.set_ylabel("Fraction of actual shifters")
ax.set_title("Calibration: baseline vs. XGBoost")
ax.legend()
plt.tight_layout()
plt.savefig("calibration_curve.png", dpi=200)
plt.close()
print("\nSaved calibration_curve.png")

# --- Confusion matrix plot (XGBoost) ---
cm = confusion_matrix(y_test, xgb_pred)
disp = ConfusionMatrixDisplay(cm, display_labels=["Stable", "Shifter"])
disp.plot(cmap="Blues")
plt.title("XGBoost confusion matrix (test set)")
plt.tight_layout()
plt.savefig("confusion_matrix_xgb.png", dpi=200)
plt.close()
print("Saved confusion_matrix_xgb.png")

# --- Subgroup-level evaluation: older (65+) vs. younger respondents ---

older_mask_eval = X_test["age"] >= 65
for label, mask in [("65+", older_mask_eval), ("under 65", ~older_mask_eval)]:
    if mask.sum() > 0:
        sub_f1 = f1_score(y_test[mask], xgb_pred[mask])
        sub_report = classification_report(y_test[mask], xgb_pred[mask], output_dict=True)
        print(f"\nXGBoost, {label} (N={mask.sum()}): F1={sub_f1:.3f}, "
              f"shifter precision={sub_report['1']['precision']:.3f}, "
              f"shifter recall={sub_report['1']['recall']:.3f}")


def find_best_threshold(y_true, y_proba):
    """Sweep thresholds 0.05-0.95 and return the one maximizing F1."""
    thresholds = np.arange(0.05, 0.96, 0.05)
    f1s = [f1_score(y_true, (y_proba >= t).astype(int)) for t in thresholds]
    best_i = int(np.argmax(f1s))
    return thresholds[best_i], f1s[best_i]


# ==============================================================================
# THRESHOLD TUNING
# Both models above use a flat 0.5 cutoff, which (see the confusion matrix)
# produces far more false positives than true positives. The threshold is
# tuned here using TRAINING-set predictions only, then applied once to the
# test set 
# ==============================================================================
xgb_oof_proba = cross_val_predict(
    best_model, X_train, y_train, cv=cv, method="predict_proba"
)[:, 1]
xgb_best_threshold, xgb_best_train_f1 = find_best_threshold(y_train, xgb_oof_proba)
xgb_tuned_pred = (xgb_pred_proba >= xgb_best_threshold).astype(int)
xgb_tuned_f1 = f1_score(y_test, xgb_tuned_pred)
print(f"\nXGBoost tuned threshold: {xgb_best_threshold:.2f} "
      f"(out-of-fold train F1: {xgb_best_train_f1:.3f})")
print(f"XGBoost F1 at tuned threshold (test set): {xgb_tuned_f1:.3f} "
      f"(vs. {xgb_f1:.3f} at default 0.5)")

logit_train_pred_proba = logit_model.predict(X_train_logit)
logit_best_threshold, logit_best_train_f1 = find_best_threshold(
    y_train_logit, logit_train_pred_proba
)
logit_tuned_pred = (logit_pred_proba >= logit_best_threshold).astype(int)
logit_tuned_f1 = f1_score(y_test_logit, logit_tuned_pred)
print(f"\nLogit tuned threshold: {logit_best_threshold:.2f} "
      f"(in-sample train F1: {logit_best_train_f1:.3f})")
print(f"Logit F1 at tuned threshold (test set): {logit_tuned_f1:.3f} "
      f"(vs. {logit_f1:.3f} at default 0.5)")

# ==============================================================================
# SHAP INTERPRETATION
# ==============================================================================
explainer = shap.TreeExplainer(best_model)
shap_values = explainer.shap_values(X_test)

plt.figure()
shap.summary_plot(shap_values, X_test, show=False)
plt.tight_layout()
plt.savefig("shap_beeswarm.png", dpi=200)
plt.close()
print("\nSaved shap_beeswarm.png")

plt.figure()
shap.dependence_plot(CYNICISM_COL, shap_values, X_test, interaction_index="age", show=False)
plt.tight_layout()
plt.savefig("shap_cynicism_dependence.png", dpi=200)
plt.close()
print("Saved shap_cynicism_dependence.png (cynicism, colored by age)")

plt.figure()
shap.dependence_plot("age", shap_values, X_test, interaction_index=CYNICISM_COL, show=False)
plt.tight_layout()
plt.savefig("shap_age_dependence.png", dpi=200)
plt.close()
print("Saved shap_age_dependence.png (age, colored by cynicism)")

mean_abs_shap = np.abs(shap_values).mean(axis=0)
importance = pd.Series(mean_abs_shap, index=X_test.columns).sort_values(ascending=False)
print("\nSHAP feature importance ranking (mean |SHAP value|):")
print(importance)

# ==============================================================================
# OLDER-SHIFTER SUBGROUP CHECK
# Filters the already-computed SHAP values down to respondents aged 65+
# who were predicted/actual shifters
# ==============================================================================
older_shifter_mask = (X_test["age"] >= 65) & (y_test.values == 1)
print(f"\nOlder shifters in test set: {older_shifter_mask.sum()}")
if older_shifter_mask.sum() > 0:
    older_importance = pd.Series(
        np.abs(shap_values[older_shifter_mask]).mean(axis=0), index=X_test.columns
    ).sort_values(ascending=False)
    print("SHAP feature importance ranking, older shifters only:")
    print(older_importance)

# ==============================================================================
# SHAP INTERACTION VALUES
# ==============================================================================
interaction_values = explainer.shap_interaction_values(X_test)
cyn_idx = X_test.columns.get_loc(CYNICISM_COL)
age_idx = X_test.columns.get_loc("age")


cyn_age_interaction = (
    interaction_values[:, cyn_idx, age_idx] + interaction_values[:, age_idx, cyn_idx]
)
mean_abs_interaction = np.abs(cyn_age_interaction).mean()
mean_abs_cyn_main = np.abs(interaction_values[:, cyn_idx, cyn_idx]).mean()
mean_abs_age_main = np.abs(interaction_values[:, age_idx, age_idx]).mean()
print(f"\nCynicism x age interaction strength (mean |value|): {mean_abs_interaction:.4f}")
print(f"  (vs. cynicism main effect: {mean_abs_cyn_main:.4f}, "
      f"age main effect: {mean_abs_age_main:.4f})")

# ==============================================================================
# ERROR ANALYSIS
# ==============================================================================
error_df = X_test.copy()
error_df["actual"] = y_test.values
error_df["predicted"] = xgb_pred
error_df["outcome"] = np.select(
    [
        (error_df["actual"] == 1) & (error_df["predicted"] == 1),
        (error_df["actual"] == 0) & (error_df["predicted"] == 0),
        (error_df["actual"] == 0) & (error_df["predicted"] == 1),
        (error_df["actual"] == 1) & (error_df["predicted"] == 0),
    ],
    ["true_positive", "true_negative", "false_positive", "false_negative"],
    default="unknown",
)
print("\nMean feature values by prediction outcome:")
print(error_df.groupby("outcome")[[CYNICISM_COL, "age", "political_interest_reversed"]].mean().round(3))