import os
os.chdir('C:/Users/Lucas/UNIklas/')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyreadstat
import seaborn as sns

# ==============================================================================
# DATASET
# ==============================================================================
df, meta = pyreadstat.read_sav("DPES2023DATASET.sav")
print(
    f"Aantal respondenten (oorspronkelijk): {df.shape[0]}, Aantal variabelen: {df.shape[1]}"
)

# ==============================================================================
# POLITICAL CYNICISM INDEX (6-item scale)
# Items: V252, V253, V326, V261, V262, V322
# ==============================================================================
df_cyn = df.copy()

# 1. Binary cynicism-worded items (1 = True/Cynical -> 1.0 | 2 = Not true -> 0.0)
#    "MPs do not care about people like me" (V252)
#    "Parties only interested in my vote, not my opinion" (V253)
binary_items = ["V252", "V253"]
for col in binary_items:
    valid = df_cyn[col].apply(lambda x: x if x in [1, 2] else np.nan)
    df_cyn[f"{col}_norm"] = valid.map({1.0: 1.0, 2.0: 0.0})

# 2. Trust-worded items, 5-point agree-disagree (5 = Fully disagree = Maximum cynical)
#    "Politicians keep their promises" (V262), "Most politicians are trustworthy" (V322)
pos_items = ["V262", "V322"]
for col in pos_items:
    valid = df_cyn[col].apply(lambda x: x if x in [1, 2, 3, 4, 5] else np.nan)
    df_cyn[f"{col}_norm"] = (valid - 1) / 4.0

# 3. Cynicism-worded items, 5-point agree-disagree (1 = Fully/Strongly agree = Maximum cynical -> inverse)
#    "Politicians are profiteers" (V261)
#    "Politicians care only about interests of rich and powerful" (V326)
neg_items = ["V261", "V326"]
for col in neg_items:
    valid = df_cyn[col].apply(lambda x: x if x in [1, 2, 3, 4, 5] else np.nan)
    df_cyn[f"{col}_norm"] = (5 - valid) / 4.0

norm_cols = [
    f"{col}_norm"
    for col in ["V252", "V253", "V326", "V261", "V262", "V322"]
]

for col in norm_cols:
    df[col] = df_cyn[col]

df["political_cynicism_index"] = df[norm_cols].mean(axis=1, skipna=False)

# --- Marginal distribution of the cynicism index itself (not split by
# target) 
from scipy.stats import skew

cyn_index_valid = df["political_cynicism_index"].dropna()
print(f"\nPolitical cynicism index -- marginal distribution (N={len(cyn_index_valid)}):")
print(f"  mean   = {cyn_index_valid.mean():.3f}")
print(f"  median = {cyn_index_valid.median():.3f}")
print(f"  std    = {cyn_index_valid.std():.3f}")
print(f"  skew   = {skew(cyn_index_valid):.3f}")
print(f"  min/max = {cyn_index_valid.min():.3f} / {cyn_index_valid.max():.3f}")
pct_at_floor = (cyn_index_valid == 0).mean() * 100
pct_at_ceiling = (cyn_index_valid == 1).mean() * 100
print(f"  % exactly at floor (0.0) = {pct_at_floor:.1f}%")
print(f"  % exactly at ceiling (1.0) = {pct_at_ceiling:.1f}%")


n_distinct = cyn_index_valid.round(4).nunique()
print(f"\nNumber of distinct possible index values observed: {n_distinct}")

# ==============================================================================
# CRONBACH'S ALPHA = measures how trustworthy/internally consistent a set of
# questions measures one underlying construct.
# Computed here on ALL respondents with complete data on the 6 items, BEFORE
# any filtering to the target-variable sample
# ==============================================================================

def cronbach_alpha(df_items):
    items_matrix = df_items.dropna()
    k = items_matrix.shape[1]
    item_variances = items_matrix.var(axis=0, ddof=1).sum()
    total_variance = items_matrix.sum(axis=1).var(ddof=1)
    return (k / (k - 1)) * (1 - (item_variances / total_variance)), items_matrix.shape[0]

alpha_score, alpha_n = cronbach_alpha(df[norm_cols])
print(
    f"\nCronbach's Alpha of Political Cynicism Index (N={alpha_n}): "
    f"{alpha_score:.3f} (goal: >= 0.70)"
)

# --- Straightlining check 
cyn_items_complete = df[norm_cols].dropna()
straightliners = (cyn_items_complete.nunique(axis=1) == 1).sum()
print(
    f"Respondents giving an identical response to all 6 cynicism items: "
    f"{straightliners} out of {len(cyn_items_complete)} with complete data "
    f"({100*straightliners/len(cyn_items_complete):.1f}%)"
)

# ==============================================================================
# TARGET VARIABLE  (target_shifter)
# ==============================================================================
intent_col = "N76"  # Stemintentie voorlijst 2023
vote_col = "V163"  # Daadwerkelijke stem nalijst

invalid_intent_codes = [96, 97, 98, 993, 995, 999]
invalid_vote_codes = [30, 31, 993, 994, 995, 999]

# Filter ongeldige stemrespons
df_clean = df[
    ~df[intent_col].isin(invalid_intent_codes)
    & ~df[vote_col].isin(invalid_vote_codes)
].copy()

# only keep people that answered both questions N76/V163
df_clean = df_clean.dropna(subset=[intent_col, vote_col]).copy()

# Target: 1 = Shifter, 0 = Stabiel
df_clean["target_shifter"] = (
    df_clean[intent_col] != df_clean[vote_col]
).astype(int)

print(
    f"\nAantal overgebleven respondenten met geldige partijkeuze: {len(df_clean)}"
)
print("\nVerdeling van target_shifter (%):")
print(
    (df_clean["target_shifter"].value_counts(normalize=True) * 100).round(2)
)

plt.figure(figsize=(7, 5))
ax = sns.countplot(
    data=df_clean,
    x="target_shifter",
    hue="target_shifter",
    palette=["navy", "crimson"],  # Index 0 = Navy, Index 1 = Crimson
    legend=False,
)

# ==============================================================================
# DEMOGRAPHIC VARIABLES
# ==============================================================================
# Gender (V010)
gender_map = {1: "Man", 2: "Woman", 3: "Other"}
df_clean["gender_label"] = df_clean["V010"].map(gender_map)


df_clean["is_woman"] = df_clean["V010"].map({1: 0, 2: 1, 3: np.nan})

# age (V012)
df_clean["age"] = df_clean["V012"].replace(999, np.nan)
df_clean["age_group"] = pd.cut(
    df_clean["age"],
    bins=[17, 29, 49, 64, 100],
    labels=["18-29", "30-49", "50-64", "65+"],
)

# income (N92)
df_clean["income_clean"] = df_clean["N92"].replace([994, 995, 999], np.nan)
cond_inc = [
    df_clean["income_clean"].between(1, 5),
    df_clean["income_clean"].between(6, 10),
    df_clean["income_clean"].between(11, 15),
]
df_clean["income_class"] = np.select(cond_inc, [1, 2, 3], default=np.nan)

# education (V368_REC) 
edu_clean = df_clean["V368_recode"].replace([7, 8, 9, 994, 995, 999], np.nan)
cond_edu = [
    edu_clean.between(1, 2),   # Basisonderwijs, Vmbo/Mavo
    edu_clean.between(3, 4),   # Mbo, Havo/Vwo
    edu_clean.between(5, 6),   # Hbo, WO
]
df_clean["edu_class"] = np.select(cond_edu, [1, 2, 3], default=np.nan)

# political interest (S043)
df_clean["political_interest"] = df_clean["S043"].replace([994, 995, 999], np.nan)

# Reversed version for readability: S043's original coding runs 1=very
# interested to 4=not at all interested
df_clean["political_interest_reversed"] = 5 - df_clean["political_interest"]

# degree of urbanization (Urbanity) 
df_clean["urbanization"] = df_clean["Urbanity"]

# marital status (N94)
df_clean["marital_status"] = df_clean["N94"].replace([995, 999], np.nan)

# religious denomination (V351)
df_clean["religion"] = df_clean["V351"].replace([994, 995, 999], np.nan)

# ==============================================================================
# joint completeness across cynicism scale + all 7 controls, and the resulting 
# class balance on target_shifter within that sample. (just used to check)
# ==============================================================================
control_cols = [
    "political_interest", "urbanization", "is_woman", "edu_class",
    "income_class", "marital_status", "religion",
]
final_complete_mask = df_clean[norm_cols + control_cols].notna().all(axis=1)
df_final = df_clean[final_complete_mask].copy()

print(
    f"\nFinal complete-case N (cynicism scale + all 7 controls): "
    f"{len(df_final)} (out of {len(df_clean)} with a valid target)"
)
print("Class balance of target_shifter in this final sample (%):")
print((df_final["target_shifter"].value_counts(normalize=True) * 100).round(2))

# ==============================================================================
# VISUALISATIONS
# ==============================================================================

plt.figure(figsize=(8, 5))
sns.kdeplot(
    data=df_clean,
    x="political_cynicism_index",
    hue="target_shifter",
    common_norm=False,
    fill=True,
    alpha=0.3,
    palette={0: "navy", 1: "crimson"},
)
plt.title("Distribution political cynicism index: stable vs. shifters")
plt.xlabel("Political Cynicism Index (0.0 = Minimum, 1.0 = Maximum)")
plt.ylabel("Dichtheid (Density)")
plt.legend(["Shifters (1)", "Stable (0)"])
plt.tight_layout()
plt.show()


plt.figure(figsize=(7, 5))
sns.violinplot(
    data=df_clean,
    x="target_shifter",
    y="political_cynicism_index",
    hue="target_shifter",
    palette={0: "navy", 1: "crimson"},
    legend=False,
    inner="quartile",
)
plt.title("Political cynicism index by shifter status")
plt.xlabel("target_shifter (0 = Stable, 1 = Shifter)")
plt.ylabel("Political Cynicism Index")
plt.tight_layout()
plt.show()

# CORRELATION MATRIX
# Gender uses "is_woman" (a binary  0/1 indicator) since a binary variable's correlation
# with a continuous one is a legitimate point-biserial correlation, unlike
# raw V010's meaningless 1/2/3 ordering. 
# Political interest uses the reversed version so higher always means "more interested" here, matching
# the intuitive direction of every other variable in this matrix 

analysis_cols = norm_cols + [
    "political_cynicism_index",
    "age",
    "edu_class",
    "income_class",
    "political_interest_reversed",
    "urbanization",
    "is_woman",
    "marital_status",
    "religion",
    "target_shifter",
]

plt.figure(figsize=(14, 11))
sns.heatmap(
    df_clean[analysis_cols].corr(),
    annot=True,
    fmt=".2f",
    cmap="coolwarm",
    vmin=-1,
    vmax=1,
)
plt.title("Correlation matrix cynicism items & demographic characteristics")
plt.tight_layout()
plt.show()

# Demographic characteristics Shifters
fig, axes = plt.subplots(1, 3, figsize=(16, 4))

# Shifters per education level
sns.barplot(
    data=df_clean,
    x="edu_class",
    y="target_shifter",
    ax=axes[0],
    hue="edu_class",
    palette="Blues_d",
    legend=False,
    errorbar=None,
)
axes[0].set_xticklabels(["Low", "Middle", "High"])
axes[0].set_title("% Shifters per education level")
axes[0].set_ylabel("Proportion shifters")
axes[0].set_xlabel("education level")

# Shifters per age group
sns.barplot(
    data=df_clean,
    x="age_group",
    y="target_shifter",
    ax=axes[1],
    hue="age_group",
    palette="Greens_d",
    legend=False,
    errorbar=None,
)
axes[1].set_title("% Shifters per age category")
axes[1].set_xlabel("age category")
axes[1].set_ylabel("")

# Shifters per gender
sns.barplot(
    data=df_clean,
    x="gender_label",
    y="target_shifter",
    ax=axes[2],
    hue="gender_label",
    palette="Purples_d",
    legend=False,
    errorbar=None,
)
axes[2].set_title("% Shifters per gender")
axes[2].set_xlabel("gender")
axes[2].set_ylabel("")

plt.tight_layout()
plt.show()

# Shifter rate across the four remaining controls 
fig, axes = plt.subplots(1, 4, figsize=(20, 4))

sns.barplot(data=df_clean, x="political_interest", y="target_shifter",
            ax=axes[0], hue="political_interest", palette="Oranges_d",
            legend=False, errorbar=None)
axes[0].set_title("% Shifters by political interest")
axes[0].set_xlabel("political interest (1=very interested, 4=not at all)")
axes[0].set_ylabel("Proportion shifters")

sns.barplot(data=df_clean, x="urbanization", y="target_shifter",
            ax=axes[1], hue="urbanization", palette="Reds_d",
            legend=False, errorbar=None)
axes[1].set_title("% Shifters by urbanization")
axes[1].set_xlabel("urbanization (1=very high, 5=very low)")
axes[1].set_ylabel("")

sns.barplot(data=df_clean, x="marital_status", y="target_shifter",
            ax=axes[2], hue="marital_status", palette="Greys_d",
            legend=False, errorbar=None)
axes[2].set_title("% Shifters by marital status")
axes[2].set_xlabel("marital status (1=married,2=divorced,3=widowed,4=never married)")
axes[2].set_ylabel("")

sns.barplot(data=df_clean, x="religion", y="target_shifter",
            ax=axes[3], hue="religion", palette="YlOrBr_d",
            legend=False, errorbar=None)
axes[3].set_title("% Shifters by religious denomination")
axes[3].set_xlabel("religion (0=none,1=Christian,2=Islam,...)")
axes[3].set_ylabel("")

plt.tight_layout()
plt.show()

# --- Age x cynicism interaction (SRQ3)
df_clean["cynicism_tercile"] = pd.qcut(
    df_clean["political_cynicism_index"], 3, labels=["Low", "Medium", "High"]
)
interaction_table = df_clean.pivot_table(
    values="target_shifter", index="age_group", columns="cynicism_tercile",
    observed=False,
)
print("\nMean shift rate by age group x cynicism tercile:")
print(interaction_table.round(3))

plt.figure(figsize=(7, 5))
sns.heatmap(interaction_table, annot=True, fmt=".2f", cmap="coolwarm", vmin=0, vmax=interaction_table.values.max()*1.1)
plt.title("Shift rate by age group x cynicism tercile")
plt.xlabel("Cynicism tercile")
plt.ylabel("Age group")
plt.tight_layout()
plt.show()

# --- Party-switch flow among shifters: origin party (N76) -> destination
# party (V163). 
party_labels = {
    1: "VVD", 2: "D66", 3: "PvdA/GL", 4: "PVV", 5: "CDA", 6: "SP", 7: "FvD",
    8: "PvdD", 9: "CU", 10: "Volt", 11: "JA21", 12: "SGP", 13: "DENK",
    14: "50Plus", 15: "BBB", 16: "Bij1", 17: "NSC", 18: "BVNL",
}
shifters = df_clean[df_clean["target_shifter"] == 1].copy()
shifters["origin_party"] = shifters[intent_col].map(party_labels).fillna("Other")
shifters["dest_party"] = shifters[vote_col].map(party_labels).fillna("Other")

flow_table = pd.crosstab(shifters["origin_party"], shifters["dest_party"])
# keep the biggest rows/columns readable; collapse rare origin/destination
# parties into "Other" so the heatmap isn't dominated by near-empty cells
top_origins = shifters["origin_party"].value_counts().nlargest(8).index
top_dests = shifters["dest_party"].value_counts().nlargest(8).index
flow_table_top = flow_table.reindex(index=top_origins, columns=top_dests, fill_value=0)

plt.figure(figsize=(9, 7))
sns.heatmap(flow_table_top, annot=True, fmt="d", cmap="Blues")
plt.title("Party-switch flow among shifters (origin -> destination)")
plt.xlabel("Actual vote (V163)")
plt.ylabel("Pre-election intention (N76)")
plt.tight_layout()
plt.show()

################################################################################################
# Missing value check for key filtered variables

key_vars_missing_codes = {
    "N76": [96, 97, 98, 993, 995, 999],
    "V163": [30, 31, 993, 994, 995, 999],
    "V012": [999],
    "N92": [994, 995, 999],
    "V368_recode": [994, 995, 999],
    "S043": [994, 995, 999],
    "Urbanity": [999],
    "N94": [995, 999],
    "V351": [994, 995, 999],
}

print("\nMissing value check (system-missing/blank + coded missing):")
missing_pct = {}
for col, miss_codes in key_vars_missing_codes.items():
    sysmiss = df[col].isna().sum()
    coded_missing = df[col].isin(miss_codes).sum()
    total_missing = sysmiss + coded_missing
    valid = len(df) - total_missing
    missing_pct[col] = 100 * total_missing / len(df)
    print(
        f"  {col:10s} valid={valid:5d}  sysmiss(blank)={sysmiss:5d}  "
        f"coded_missing={coded_missing:5d}  total_missing={total_missing:5d}"
    )

# --- Missingness overview chart
plt.figure(figsize=(9, 5))
missing_series = pd.Series(missing_pct).sort_values(ascending=False)
sns.barplot(x=missing_series.values, y=missing_series.index, color="steelblue")
plt.title("Missingness by key variable (% system-missing + coded missing)")
plt.xlabel("% missing")
plt.ylabel("")
plt.tight_layout()
plt.show()

# ---target_shifter against the self-reported decision-timing item
# (V165, "when did you decide to vote for this party?")
v165_map = {
    1: "Election day", 2: "Last days before", 3: "Last weeks before",
    4: "A few months before", 5: "Longer beforehand",
}
df_clean["decision_timing"] = df_clean["V165"].map(v165_map)

validation_table = pd.crosstab(
    df_clean["target_shifter"], df_clean["decision_timing"], normalize="index"
) * 100
print("\nSelf-reported decision timing (%) by target_shifter (validation check):")
print(validation_table.round(1))

plt.figure(figsize=(9, 5))
validation_table.T.plot(kind="bar", ax=plt.gca(), color={0: "navy", 1: "crimson"})
plt.title("Self-reported decision timing by target_shifter (validation)")
plt.xlabel("When did you decide to vote for this party? (V165)")
plt.ylabel("% within group")
plt.legend(["Stable (0)", "Shifters (1)"])
plt.xticks(rotation=30, ha="right")
plt.tight_layout()
plt.show()