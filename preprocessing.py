"""
Shared preprocessing module for the DPES 2023 late-decider thesis.

Builds the feature set and target variable: the 6-item political cynicism
index, age (continuous, plus a 7-band Proulx-et-al.-style categorical
version for the baseline model), target_shifter, and 7 control
variables (political interest, urbanization, gender, education, income,
marital status, religion, social media use, leader-vs-program importance, 
party identification strength, left-right self-placement and sympathy for
the leader of the respondent's own intended party).

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
    
    # Prorated scale score: average whatever items are present, but only for
    # respondents who answered at least MIN_VALID_CYNICISM_ITEMS of the 6 
    # This recovers a chunk of the complete-case N lost to the previous 
    # all-6-required rule (skipna=False), without diluting the scale for 
    # respondents who barelyanswered it. 
    MIN_VALID_CYNICISM_ITEMS = 4  # out of 6
    n_valid_items = df[norm_cols].notna().sum(axis=1)
    cynicism_mean = df[norm_cols].mean(axis=1, skipna=True)
    df["political_cynicism_index"] = cynicism_mean.where(
        n_valid_items >= MIN_VALID_CYNICISM_ITEMS, np.nan
    )

    # Diagnostic: how many respondents are recovered at each threshold, vs.
    # the all-6-items 
    print("Cynicism index: respondents with >=N of 6 items present:")
    for n in range(6, 0, -1):
        print(f"  >={n} items: {(n_valid_items >= n).sum()}")

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
    # CONTROL VARIABLES (9) -- Proulx et al. (2025)'s original 7, plus some
    # added for this thesis.
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

    # --- Social media use (Q02f) ---
    # Raw 0-7 (days/week social media was used to follow party/candidate
    # news during the campaign)
    df_clean["social_media_use"] = df_clean["Q02f"].replace([993, 994, 995, 999], np.nan)

    # --- Leader vs. program importance (P55) ---
    # Raw 0-10 (0 = party leaders much more important, 10 = party programs
    # much more important)
    df_clean["leader_program_importance"] = df_clean["P55"].replace(
        [993, 994, 995, 999], np.nan
    )

    # ==========================================================================
    # NEW CANDIDATE VARIABLES 
    # ==========================================================================

    # --- Party identification strength (S110 + S111) ---
    # S110: "Do you consider yourself an adherent of a political party?"
    # (1=yes, 2=no). S111: "Would you consider yourself a strongly convinced /
    # convinced / not so convinced adherent of this party?" (1=very convinced,
    # 2=convinced, 3=not so convinced). Combined into one 0-3 ordinal
    # scale, higher = stronger identification:
    #   0 = not a party adherent at all (S110 == 2)
    #   1 = adherent, "not so convinced"   (S111 == 3)
    #   2 = adherent, "convinced"          (S111 == 2)
    #   3 = adherent, "very convinced"     (S111 == 1)
    s110 = df_clean["S110"].replace([994, 995, 999], np.nan)
    s111 = df_clean["S111"].replace([993, 994, 995, 999], np.nan)
    df_clean["party_id_strength"] = np.select(
        [s110 == 2, s111 == 3, s111 == 2, s111 == 1],
        [0, 1, 2, 3],
        default=np.nan,
    )

    # --- Left-right self-placement (V133) ---
    df_clean["left_right_self"] = df_clean["V133"].replace(
        [11, 993, 994, 995, 999], np.nan
    )

    # --- Sympathy for the leader of the respondent's own intended party ---
    # Party-leader sympathy (0-10 "how sympathetic do you find this leader")
    # is asked once per leader, in 17 separate columns (one per party).
    # Mapping each respondent's own pre-election vote intention (N76) to the
    # matching leader-sympathy column, built by matching the codebook's N76
    # party-code list against its leader-sympathy variable/name lists
    # positionally (both lists are given in the same party order):
    #   V213=Yesilgöz(VVD) V218=Jetten(D66) P52=Timmermans(PvdA/GL)
    #   V215=Wilders(PVV) V216=Bontenbal(CDA) V217=Marijnissen(SP)
    #   V225=Baudet(FvD) V222=Ouwehand(PvdD) V219=Bikker(CU)
    #   N145=Dassen(Volt) N144=Eerdmans(JA21) V221=Stoffer(SGP)
    #   V224=van Baarle(DENK) N146=van der Plas(BBB) N147=Olf(Bij1)
    #   P53=Omtzigt(NSC) P54=van Haga(BVNL)
    _leader_sympathy_var = {
        1: "V213", 2: "V218", 3: "P52", 4: "V215", 5: "V216", 6: "V217",
        7: "V225", 8: "V222", 9: "V219", 10: "N145", 11: "N144", 12: "V221",
        13: "V224", 15: "N146", 16: "N147", 17: "P53", 18: "P54",
    }
    leader_sympathy_values = pd.Series(np.nan, index=df_clean.index, dtype=float)
    intended_party = df_clean["N76"]
    for party_code, varname in _leader_sympathy_var.items():
        mask = intended_party == party_code
        if mask.any():
            raw = df_clean.loc[mask, varname].replace([11, 994, 995, 999], np.nan)
            leader_sympathy_values.loc[mask] = raw
    df_clean["leader_sympathy_own_party"] = leader_sympathy_values

    return df_clean, norm_cols


CYNICISM_COL = "political_cynicism_index"
TARGET_COL = "target_shifter"
CONTROL_COLS = [
    "political_interest", "urbanization", "is_woman",
    "edu_class", "income_class", "marital_status", "religion",
    "social_media_use", "leader_program_importance",
    "party_id_strength", "left_right_self", "leader_sympathy_own_party",
]