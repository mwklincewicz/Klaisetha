"""
Shared preprocessing module for the DPES 2023 late-decider thesis.

Builds the feature set and target variable: the 6-item political cynicism
index, age (continuous, plus a 7-band Proulx-et-al.-style categorical
version for the baseline model), target_shifter, and 7 control
variables (political interest, urbanization, gender, education, income,
marital status, religion).

"""
import numpy as np
import pandas as pd
import pyreadstat


def load_and_prepare(sav_path="DPES2023DATASET.sav"):
    df, meta = pyreadstat.read_sav(sav_path)

    # ==========================================================================
    # POLITICAL CYNICISM INDEX (6-item scale)
    # (Cronbach's alpha = 0.805, N = 2,865).
    # ==========================================================================
    binary_items = ["V252", "V253"]
    for col in binary_items:
        valid = df[col].apply(lambda x: x if x in [1, 2] else np.nan)
        df[f"{col}_norm"] = valid.map({1.0: 1.0, 2.0: 0.0})

    pos_items = ["V262", "V322"]
    for col in pos_items:
        valid = df[col].apply(lambda x: x if x in [1, 2, 3, 4, 5] else np.nan)
        df[f"{col}_norm"] = (valid - 1) / 4.0

    neg_items = ["V261", "V326"]
    for col in neg_items:
        valid = df[col].apply(lambda x: x if x in [1, 2, 3, 4, 5] else np.nan)
        df[f"{col}_norm"] = (5 - valid) / 4.0

    norm_cols = [f"{c}_norm" for c in ["V252", "V253", "V326", "V261", "V262", "V322"]]
    # skipna=False: a respondent needs all 6 items or the index is NaN --
    # keeps this consistent with the population the alpha check used.
    df["political_cynicism_index"] = df[norm_cols].mean(axis=1, skipna=False)

    # ==========================================================================
    # TARGET VARIABLE: target_shifter
    # ==========================================================================
    intent_col, vote_col = "N76", "V163"
    invalid_intent_codes = [96, 97, 98, 993, 995, 999]
    invalid_vote_codes = [30, 31, 993, 994, 995, 999]

    df_clean = df[
        ~df[intent_col].isin(invalid_intent_codes)
        & ~df[vote_col].isin(invalid_vote_codes)
    ].copy()
    df_clean = df_clean.dropna(subset=[intent_col, vote_col]).copy()
    df_clean["target_shifter"] = (df_clean[intent_col] != df_clean[vote_col]).astype(int)

    # ==========================================================================
    # AGE -- a primary variable
    # Continuous version for XGBoost, which can learn a non-linear age
    # effect directly; a 7-band Proulx-et-al.-style categorical version for
    # the logit baseline, which needs explicit bins for the same purpose.
    # ==========================================================================
    df_clean["age"] = df_clean["V012"].replace(999, np.nan)
    age_bins = [17, 25, 35, 45, 55, 65, 75, 120]
    age_labels = ["18-25", "26-35", "36-45", "46-55", "56-65", "66-75", "76+"]
    df_clean["age_band"] = pd.cut(df_clean["age"], bins=age_bins, labels=age_labels)

    # ==========================================================================
    # CONTROL VARIABLES (7) -- the set Proulx et al. (2025) control for.
    # ==========================================================================
    df_clean["political_interest"] = df_clean["S043"].replace([994, 995, 999], np.nan)
    # Reversed version for readability. Using this version means higher
    # always="more interested" matching every other feature.
    df_clean["political_interest_reversed"] = 5 - df_clean["political_interest"]
    df_clean["urbanization"] = df_clean["Urbanity"]
    df_clean["is_woman"] = df_clean["V010"].map({1: 0, 2: 1, 3: np.nan})

    edu_clean = df_clean["V368_recode"].replace([7, 8, 9, 994, 995, 999], np.nan)
    df_clean["edu_class"] = np.select(
        [edu_clean.between(1, 2), edu_clean.between(3, 4), edu_clean.between(5, 6)],
        [1, 2, 3], default=np.nan,
    )

    income_clean = df_clean["N92"].replace([994, 995, 999], np.nan)
    df_clean["income_class"] = np.select(
        [income_clean.between(1, 5), income_clean.between(6, 10), income_clean.between(11, 15)],
        [1, 2, 3], default=np.nan,
    )

    df_clean["marital_status"] = df_clean["N94"].replace([995, 999], np.nan)
    df_clean["religion"] = df_clean["V351"].replace([994, 995, 999], np.nan)

    # Collapse religion categories with very few respondents into a shared
    # "other" bucket (coded 99). it's specifically needed for the logit baseline.
    rare_threshold = 30
    religion_counts = df_clean["religion"].value_counts()
    rare_categories = religion_counts[religion_counts < rare_threshold].index
    df_clean["religion"] = df_clean["religion"].replace(
        {cat: 99 for cat in rare_categories}
    )

    return df_clean, norm_cols


CYNICISM_COL = "political_cynicism_index"
TARGET_COL = "target_shifter"
CONTROL_COLS = [
    "political_interest", "urbanization", "is_woman",
    "edu_class", "income_class", "marital_status", "religion",
]