"""
Baseline (logit) + XGBoost modeling pipeline for the DPES 2023 late-decider
thesis.
-> Requires: pip install pandas numpy pyreadstat scikit-learn statsmodels xgboost shap matplotlib
Run from the same folder as preprocessing.py and DPES2023DATASET.sav.
"""
import numpy as np
import pandas as pd
import statsmodels.api as sm
import xgboost as xgb
import shap
import matplotlib.pyplot as plt
import pickle
from sklearn.model_selection import (
    train_test_split, StratifiedKFold, RandomizedSearchCV, GridSearchCV, cross_val_predict,
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

model_df = df_clean.dropna(subset=[TARGET_COL]).copy()
print(f"Respondents with a valid target_shifter: {len(model_df)}")
print(f"Class balance:\n{model_df[TARGET_COL].value_counts(normalize=True).round(3)}")

# ==============================================================================
# TRAIN / TEST SPLIT
# Held out before any model fitting or tuning. Stratified so both sides
# keep the overall shifter rate. Touched once at final evaluation.
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
# social media use (Q02f) and leader-vs- program importance (P55) 
# Categorical variables are one-hot encoded.
#
# statsmodels requires complete rows, so the baseline necessarily runs on a 
# smaller complete-case subset of train/test 
# ==============================================================================
baseline_feature_cols = [CYNICISM_COL, "age_band"] + CONTROL_COLS


def build_logit_design(df, reference_columns=None):
    X = df[[CYNICISM_COL, "political_interest_reversed", "urbanization",
            "is_woman", "edu_class", "income_class",
            "social_media_use", "leader_program_importance",
            "party_id_strength", "left_right_self",
            "leader_sympathy_own_party"]].copy()
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

# Missingness diagnostic 
missing_rates = (
    train_df[baseline_feature_cols].isna().mean().sort_values(ascending=False)
)
print("\nMissingness rate per baseline feature (train set):")
print(missing_rates.to_string(float_format=lambda v: f"{v:.3f}"))

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
# GLM with a Binomial family and freq_weights instead. 
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
            "social_media_use", "leader_program_importance",
            "party_id_strength", "left_right_self",
            "leader_sympathy_own_party"]].copy()
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

# Hyperparameter search 
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

# 5-fold CV for tuning, within the training set 
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

xgb_train_pred = best_model.predict(X_train)
xgb_train_pred_proba = best_model.predict_proba(X_train)[:, 1]
xgb_train_f1 = f1_score(y_train, xgb_train_pred)
xgb_train_auc = roc_auc_score(y_train, xgb_train_pred_proba)
print(f"\nOverfitting check -- Train F1: {xgb_train_f1:.3f} (test: {xgb_f1:.3f})  "
      f"Train ROC-AUC: {xgb_train_auc:.3f} (test: {xgb_auc:.3f})")

# ==============================================================================
# EVALUATION: baseline vs. XGBoost
# -----> the two models run on slightly different Ns (logit's complete-case
# subset vs. XGBoost's full train/test), so this is an approximate
# comparison, not a paired one 
# ==============================================================================

# --- Precision-Recall AUC ---
logit_pr_auc = average_precision_score(y_test_logit, logit_pred_proba)
xgb_pr_auc = average_precision_score(y_test, xgb_pred_proba)

# --- Calibration (Brier score) ---

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
# logit's smaller complete-case test set. This section
# re-evaluates the same trained XGBoost model restricted to the
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

# --- Confusion matrix plot (Logit) ---
cm_logit = confusion_matrix(y_test_logit, logit_pred)
disp_logit = ConfusionMatrixDisplay(cm_logit, display_labels=["Stable", "Shifter"])
disp_logit.plot(cmap="Blues")
plt.title("Logit confusion matrix (test set)")
plt.tight_layout()
plt.savefig("confusion_matrix_logit.png", dpi=200)
plt.close()
print("Saved confusion_matrix_logit.png")

#  searches for the classification threshold that gives the best F1 score,
#  instead of just using the default 0.5 cutoff (bad cut-off with imbalanced classes).
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
# tuned here using training-set predictions only, then applied once to the
# test set.
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
# SHAP INTERACTION VALUES
# ==============================================================================
interaction_values = explainer.shap_interaction_values(X_test)
cyn_idx = X_test.columns.get_loc(CYNICISM_COL)
age_idx = X_test.columns.get_loc("age")

# contextualizes the interaction’s  importance relative to each variable’s own standalone effect.
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

# ==============================================================================
# WEIGHTED VS. UNWEIGHTED XGBOOST COMPARISON
# ==============================================================================
print("\n" + "=" * 70)
print("WEIGHTED VS. UNWEIGHTED COMPARISON")
print("=" * 70)

xgb_unweighted = xgb.XGBClassifier(
    objective="binary:logistic",
    scale_pos_weight=1.0,  
    eval_metric="logloss",
    random_state=RANDOM_STATE,
    missing=np.nan,
    **search.best_params_,
)
xgb_unweighted.fit(X_train, y_train)

xgb_unw_pred_proba = xgb_unweighted.predict_proba(X_test)[:, 1]
xgb_unw_pred = xgb_unweighted.predict(X_test)
xgb_unw_f1 = f1_score(y_test, xgb_unw_pred)
xgb_unw_auc = roc_auc_score(y_test, xgb_unw_pred_proba)
xgb_unw_pr_auc = average_precision_score(y_test, xgb_unw_pred_proba)
xgb_unw_brier = brier_score_loss(y_test, xgb_unw_pred_proba)

# Out-of-fold threshold tuning for the unweighted model
xgb_unw_oof_proba = cross_val_predict(
    xgb_unweighted, X_train, y_train, cv=cv, method="predict_proba"
)[:, 1]
xgb_unw_best_threshold, _ = find_best_threshold(y_train, xgb_unw_oof_proba)
xgb_unw_tuned_pred = (xgb_unw_pred_proba >= xgb_unw_best_threshold).astype(int)
xgb_unw_tuned_f1 = f1_score(y_test, xgb_unw_tuned_pred)

# ------------------------------------------------------------------------------
# UPDATED CALIBRATION CURVE: adds the final (unweighted) model
# ------------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(6, 6))
for name, y_true, y_proba in [
    ("Logit (weighted)", y_test_logit, logit_pred_proba),
    ("XGBoost (weighted)", y_test, xgb_pred_proba),
    ("XGBoost (final: unweighted)", y_test, xgb_unw_pred_proba),
]:
    frac_pos, mean_pred = calibration_curve(y_true, y_proba, n_bins=10)
    ax.plot(mean_pred, frac_pos, marker="o", label=name)
ax.plot([0, 1], [0, 1], linestyle="--", color="grey", label="Perfectly calibrated")
ax.set_xlabel("Mean predicted probability")
ax.set_ylabel("Fraction of actual shifters")
ax.set_title("Calibration: weighted models vs. final (unweighted) model")
ax.legend()
plt.tight_layout()
plt.savefig("calibration_curve_final.png", dpi=200)
plt.close()
print("\nSaved calibration_curve_final.png (adds the final unweighted model's calibration)")

# ==============================================================================
# MEDIAN-AGE-SPLIT ROBUSTNESS CHECK
##########################################
print("\n" + "=" * 70)
print("MEDIAN-AGE-SPLIT ROBUSTNESS CHECK (Proulx-et-al.-style subsample comparison)")
print("=" * 70)


def build_logit_design_median_split(df, reference_columns=None):
    """Same control set as build_logit_design, but WITHOUT age_band dummies
    -- see comment above on why age_band is dropped for this specific
    within-age-half comparison."""
    X = df[[CYNICISM_COL, "political_interest_reversed", "urbanization",
            "is_woman", "edu_class", "income_class",
            "social_media_use", "leader_program_importance",
            "party_id_strength", "left_right_self",
            "leader_sympathy_own_party"]].copy()
    X = pd.concat([
        X,
        pd.get_dummies(df["marital_status"], prefix="marital", drop_first=True),
        pd.get_dummies(df["religion"], prefix="religion", drop_first=True),
    ], axis=1)
    X = sm.add_constant(X, has_constant="add")
    X = X.astype(float)
    if reference_columns is not None:
        X = X.reindex(columns=reference_columns, fill_value=0.0)
    return X


# --- Logit: fit separately on each age half, full baseline sample ---
baseline_full = pd.concat([baseline_train, baseline_test])
age_median_logit = baseline_full["age"].median()
print(f"\nLogit sample median age: {age_median_logit:.1f} "
      f"(N={len(baseline_full)}, complete-case baseline sample)")

for label, half_df in [
    ("Below median age", baseline_full[baseline_full["age"] < age_median_logit]),
    ("At/above median age", baseline_full[baseline_full["age"] >= age_median_logit]),
]:
    X_half = build_logit_design_median_split(half_df)
    y_half = half_df[TARGET_COL]
    half_weight = (y_half == 0).sum() / max((y_half == 1).sum(), 1)
    w_half = np.where(y_half == 1, half_weight, 1.0)
    try:
        half_model = sm.GLM(
            y_half, X_half, family=sm.families.Binomial(), freq_weights=w_half,
        ).fit()
        cyn_coef = half_model.params[CYNICISM_COL]
        cyn_se = half_model.bse[CYNICISM_COL]
        cyn_p = half_model.pvalues[CYNICISM_COL]
        print(f"\n{label} (N={len(half_df)}, shift rate={y_half.mean():.3f}):")
        print(f"  Cynicism coefficient: {cyn_coef:.3f}  (SE={cyn_se:.3f}, p={cyn_p:.3f})")
    except Exception as e:
        print(f"\n{label} (N={len(half_df)}): GLM fit failed -- {e}")


# --- XGBoost: median threshold from train, applied to split test set ---
age_median_xgb = X_train["age"].median()
print(f"\nXGBoost median age (from train set): {age_median_xgb:.1f}")

for label, mask in [
    ("Below median age", X_test["age"] < age_median_xgb),
    ("At/above median age", X_test["age"] >= age_median_xgb),
]:
    if mask.sum() > 0:
        sub_report = classification_report(
            y_test[mask], xgb_unw_tuned_pred[mask], output_dict=True, zero_division=0
        )
        sub_f1 = f1_score(y_test[mask], xgb_unw_tuned_pred[mask])
        print(f"\n{label} (N={mask.sum()}): F1={sub_f1:.3f}  "
              f"precision={sub_report['1']['precision']:.3f}  "
              f"recall={sub_report['1']['recall']:.3f}")
        half_shap_cyn = np.abs(shap_values[mask.values, X_test.columns.get_loc(CYNICISM_COL)]).mean()
        print(f"  Mean |SHAP| for cynicism in this half: {half_shap_cyn:.4f}")

# ------------------------------------------------------------------------------
# VISUALIZING THE MEDIAN-AGE SPLIT ON THE CYNICISM DEPENDENCE PLOT
# ------------------------------------------------------------------------------
below_median_age = (X_test["age"] < age_median_xgb).values
plt.figure(figsize=(7, 5))
plt.scatter(
    X_test.loc[below_median_age, CYNICISM_COL],
    shap_values[below_median_age, cyn_idx],
    color="tab:blue", alpha=0.6, s=18,
    label=f"Below median age (<{age_median_xgb:.0f})",
)
plt.scatter(
    X_test.loc[~below_median_age, CYNICISM_COL],
    shap_values[~below_median_age, cyn_idx],
    color="tab:red", alpha=0.6, s=18,
    label=f"At/above median age (≥{age_median_xgb:.0f})",
)
plt.xlabel("political_cynicism_index")
plt.ylabel("SHAP value for political_cynicism_index")
plt.title("Cynicism SHAP value, split at median age")
plt.legend()
plt.tight_layout()
plt.savefig("shap_cynicism_median_age_split.png", dpi=200)
plt.close()
print("Saved shap_cynicism_median_age_split.png (two-color version of the median-age split)")

# ------------------------------------------------------------------------------
#  AGE-BINNED SHAP TREND (sextiles)
# ------------------------------------------------------------------------------
age_bins_cont = pd.qcut(X_test["age"], q=6, duplicates="drop")
shap_cyn_values = shap_values[:, cyn_idx]
age_bin_df = pd.DataFrame({
    "age_bin": age_bins_cont,
    "abs_shap_cyn": np.abs(shap_cyn_values),
})
age_bin_summary = age_bin_df.groupby("age_bin", observed=True)["abs_shap_cyn"].agg(["mean", "sem", "count"])
print("\nMean |SHAP| for cynicism by age bin (sextiles):")
print(age_bin_summary)

plt.figure(figsize=(7, 5))
bin_positions = range(len(age_bin_summary))
plt.errorbar(
    bin_positions, age_bin_summary["mean"], yerr=age_bin_summary["sem"],
    marker="o", capsize=4,
)
plt.xticks(bin_positions, [str(iv) for iv in age_bin_summary.index], rotation=30, ha="right")
plt.xlabel("Age bin")
plt.ylabel("Mean |SHAP value| for cynicism")
plt.title("Cynicism's SHAP impact across the age range")
plt.tight_layout()
plt.savefig("shap_cynicism_by_age_bin.png", dpi=200)
plt.close()
print("Saved shap_cynicism_by_age_bin.png")

# ------------------------------------------------------------------------------
# CYNICISM x AGE INTERACTION VALUE, PLOTTED AGAINST AGE DIRECTLY
# ------------------------------------------------------------------------------
plt.figure(figsize=(7, 5))
plt.scatter(X_test["age"], cyn_age_interaction, alpha=0.5, s=18, color="tab:purple")
plt.axhline(0, color="grey", linestyle="--", linewidth=1)
plt.xlabel("age")
plt.ylabel("Cynicism x age SHAP interaction value")
plt.title("Cynicism x age interaction value across the age range")
plt.tight_layout()
plt.savefig("shap_cyn_age_interaction_by_age.png", dpi=200)
plt.close()
print("Saved shap_cyn_age_interaction_by_age.png")

# ------------------------------------------------------------------------------
# PARTIAL DEPENDENCE PLOT FOR AGE
# ------------------------------------------------------------------------------
age_grid = np.linspace(X_test["age"].min(), X_test["age"].max(), 25)
pdp_values = []
for age_val in age_grid:
    X_pdp = X_test.copy()
    X_pdp["age"] = age_val
    pdp_values.append(xgb_unweighted.predict_proba(X_pdp)[:, 1].mean())

plt.figure(figsize=(7, 5))
plt.plot(age_grid, pdp_values, marker="o", markersize=4)
plt.xlabel("age")
plt.ylabel("Mean predicted probability of shifting")
plt.title("Partial dependence: age (final model)")
plt.tight_layout()
plt.savefig("pdp_age.png", dpi=200)
plt.close()
print("Saved pdp_age.png")

# ==============================================================================
# CYNICISM x POLITICAL INTEREST INTERACTION
# ==============================================================================
print("\n" + "=" * 70)
print("CYNICISM x POLITICAL INTEREST INTERACTION")
print("=" * 70)

plt.figure()
shap.dependence_plot(
    CYNICISM_COL, shap_values, X_test,
    interaction_index="political_interest_reversed", show=False,
)
plt.tight_layout()
plt.savefig("shap_cynicism_interest_dependence.png", dpi=200)
plt.close()
print("\nSaved shap_cynicism_interest_dependence.png (cynicism, colored by political interest)")

plt.figure()
shap.dependence_plot(
    "political_interest_reversed", shap_values, X_test,
    interaction_index=CYNICISM_COL, show=False,
)
plt.tight_layout()
plt.savefig("shap_interest_cynicism_dependence.png", dpi=200)
plt.close()
print("Saved shap_interest_cynicism_dependence.png (political interest, colored by cynicism)")

interest_idx = X_test.columns.get_loc("political_interest_reversed")
cyn_interest_interaction = (
    interaction_values[:, cyn_idx, interest_idx] + interaction_values[:, interest_idx, cyn_idx]
)
mean_abs_cyn_interest = np.abs(cyn_interest_interaction).mean()
mean_abs_interest_main = np.abs(interaction_values[:, interest_idx, interest_idx]).mean()
print(f"\nCynicism x political interest interaction strength (mean |value|): "
      f"{mean_abs_cyn_interest:.4f}")
print(f"  (vs. cynicism main effect: {mean_abs_cyn_main:.4f}, "
      f"political interest main effect: {mean_abs_interest_main:.4f})")
print(f"  (for reference, cynicism x age interaction strength was: "
      f"{mean_abs_interaction:.4f})")

# ==============================================================================
# CYNICISM INTERACTION SWEEP: ALL FEATURES, RANKED
# ==============================================================================
print("\n" + "=" * 70)
print("CYNICISM INTERACTION SWEEP (all features, ranked)")
print("=" * 70)

cyn_interaction_strength = {}
for other_col in X_test.columns:
    if other_col == CYNICISM_COL:
        continue
    other_idx = X_test.columns.get_loc(other_col)
    pair_interaction = (
        interaction_values[:, cyn_idx, other_idx] + interaction_values[:, other_idx, cyn_idx]
    )
    cyn_interaction_strength[other_col] = np.abs(pair_interaction).mean()

cyn_interaction_ranked = pd.Series(cyn_interaction_strength).sort_values(ascending=False)
print(f"\nCynicism main effect (reference): {mean_abs_cyn_main:.4f}\n")
print("Cynicism's interaction strength with every other feature (mean |value|):")
print(cyn_interaction_ranked)

# Features that already have a dedicated dependence plot elsewhere in this
# script -- skip those here and only plot genuinely new candidates.
already_plotted = {"age", "political_interest_reversed", "party_id_strength"}
top_new = [f for f in cyn_interaction_ranked.index if f not in already_plotted][:3]
print(f"\nGenerating dependence plots for the top new interaction partners: {top_new}")

for feat in top_new:
    plt.figure()
    shap.dependence_plot(
        CYNICISM_COL, shap_values, X_test, interaction_index=feat, show=False,
    )
    plt.tight_layout()
    fname = f"shap_cynicism_{feat}_dependence.png"
    plt.savefig(fname, dpi=200)
    plt.close()
    print(f"Saved {fname} (cynicism, colored by {feat})")

# ------------------------------------------------------------------------------
# NORMALIZED INTERACTION STRENGTH
# ------------------------------------------------------------------------------
print("\n--- Normalized by each feature's own main effect (controls for overall importance) ---")

other_main_effects = {}
for other_col in X_test.columns:
    if other_col == CYNICISM_COL:
        continue
    other_idx = X_test.columns.get_loc(other_col)
    other_main_effects[other_col] = np.abs(interaction_values[:, other_idx, other_idx]).mean()

normalized_interaction = {}
for other_col, raw_strength in cyn_interaction_strength.items():
    other_main = other_main_effects[other_col]
    denom = np.sqrt(mean_abs_cyn_main * other_main)
    normalized_interaction[other_col] = raw_strength / denom if denom > 0 else np.nan

normalized_ranked = pd.Series(normalized_interaction).sort_values(ascending=False)
print("\nCynicism's interaction strength with every other feature, normalized by\n"
      "sqrt(cynicism main effect x that feature's own main effect):")
print(normalized_ranked)

# ==============================================================================
# FEATURE PRUNING: FULL VS. REDUCED FEATURE SET
# ==============================================================================
print("\n" + "=" * 70)
print("FEATURE PRUNING: FULL VS. REDUCED FEATURE SET")
print("=" * 70)

SHAP_DROP_THRESHOLD = 0.01
low_shap_features = importance[importance < SHAP_DROP_THRESHOLD].index.tolist()
print(f"\nFeatures with mean |SHAP| < {SHAP_DROP_THRESHOLD}: {low_shap_features}")

reduced_cols = [c for c in X_train.columns if c not in low_shap_features]
X_train_reduced = X_train[reduced_cols]
X_test_reduced = X_test[reduced_cols]
print(f"Reduced feature set: {len(reduced_cols)} columns (from {X_train.shape[1]})")

xgb_reduced = xgb.XGBClassifier(
    objective="binary:logistic", scale_pos_weight=1.0, eval_metric="logloss",
    random_state=RANDOM_STATE, missing=np.nan, **search.best_params_,
)
xgb_reduced.fit(X_train_reduced, y_train)
xgb_reduced_pred_proba = xgb_reduced.predict_proba(X_test_reduced)[:, 1]
xgb_reduced_auc = roc_auc_score(y_test, xgb_reduced_pred_proba)
xgb_reduced_pr_auc = average_precision_score(y_test, xgb_reduced_pred_proba)
xgb_reduced_brier = brier_score_loss(y_test, xgb_reduced_pred_proba)

xgb_reduced_oof_proba = cross_val_predict(
    xgb_reduced, X_train_reduced, y_train, cv=cv, method="predict_proba"
)[:, 1]
xgb_reduced_best_threshold, _ = find_best_threshold(y_train, xgb_reduced_oof_proba)
xgb_reduced_tuned_pred = (xgb_reduced_pred_proba >= xgb_reduced_best_threshold).astype(int)
xgb_reduced_tuned_f1 = f1_score(y_test, xgb_reduced_tuned_pred)

print(f"\n{'Model':<30} {'N features':>10} {'F1 (tuned)':>11} {'ROC-AUC':>9} {'PR-AUC':>8} {'Brier':>8}")
print(f"{'Full feature set':<30} {X_train.shape[1]:>10} {xgb_unw_tuned_f1:>11.3f} "
      f"{xgb_unw_auc:>9.3f} {xgb_unw_pr_auc:>8.3f} {xgb_unw_brier:>8.3f}")
print(f"{'Reduced feature set':<30} {len(reduced_cols):>10} {xgb_reduced_tuned_f1:>11.3f} "
      f"{xgb_reduced_auc:>9.3f} {xgb_reduced_pr_auc:>8.3f} {xgb_reduced_brier:>8.3f}")
