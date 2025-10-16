import pandas as pd
import argparse
import os
from sklearn.metrics import confusion_matrix, accuracy_score

parser = argparse.ArgumentParser(description="Compare regulatory_according_to_rule columns and analyze matching sentences.")
parser.add_argument("-in", "--input", required=True, help="Path to CSV file containing 'regulatory_according_to_rule', 'regulatory_according_to_inlegalbert', and 'sent' columns.")
args = parser.parse_args()

csv_path = os.path.abspath(args.input)
print(f"[INFO] Reading CSV: {csv_path}")

# Load CSV
df = pd.read_csv(csv_path, low_memory=False)
print(f"[INFO] Loaded {len(df):,} rows.")
print(f"[INFO] Columns: {list(df.columns)}")

# Check required columns
cols = ["regulatory_according_to_rule", "regulatory_according_to_inlegalbert", "sent"]
missing = [c for c in cols if c not in df.columns]
if missing:
    raise SystemExit(f"[ERROR] Missing columns: {missing}")

# Sanitize
df = df[cols].copy()
#df = df.fillna({"regulatory_according_to_rule": -1, "regulatory_according_to_inlegalbert": -1, "sent": ""})
#df["regulatory_according_to_rule"] = df["regulatory_according_to_rule"].astype(int)
df["regulatory_according_to_inlegalbert"] = df["regulatory_according_to_inlegalbert"].astype(int)

# --- Subset check: are all human-regulatory_according_to_ruleed 1s also model 1s? ---
regulatory_according_to_rule_1_rows = df[df["regulatory_according_to_rule"] == 1]
subset_mask = regulatory_according_to_rule_1_rows["regulatory_according_to_inlegalbert"] == 1

true_subset_count = int(subset_mask.sum())
false_subset_count = len(subset_mask) - true_subset_count
is_subset = false_subset_count == 0

print("\n=== Subset check ===")
print(f"regulatory_according_to_rule=1 total: {len(regulatory_according_to_rule_1_rows):,}")
print(f"Those also predicted 1 by model: {true_subset_count:,}")
print(f"regulatory_according_to_rule=1 but predicted 0 by model: {false_subset_count:,}")
print(f"Are all regulatory_according_to_rule=1 rows also model=1? -> {is_subset}")

# Optional: show a few counterexamples (regulatory_according_to_rule=1, model=0)
if not is_subset and false_subset_count > 0:
    mismatches = regulatory_according_to_rule_1_rows[regulatory_according_to_rule_1_rows["regulatory_according_to_inlegalbert"] == 0]
    print("\nSample of mismatches (regulatory_according_to_rule=1, model=0):")
    for i, row in mismatches.head(5).iterrows():
        print(f"- {row['sent'][:120]}{'...' if len(row['sent']) > 120 else ''}")
    out_path = os.path.splitext(csv_path)[0] + "_regulatory_according_to_rule1_notpred1.csv"
    mismatches.to_csv(out_path, index=False)
    print(f"[INFO] Saved all {false_subset_count:,} mismatched sentences to: {out_path}")

# --- Opposite subset check: are all model-predicted 1s also regulatory_according_to_rule=1? ---
model_1_rows = df[df["regulatory_according_to_inlegalbert"] == 1]
subset_mask_model = model_1_rows["regulatory_according_to_rule"] == 1

true_subset_model_count = int(subset_mask_model.sum())
false_subset_model_count = len(subset_mask_model) - true_subset_model_count
is_subset_model = false_subset_model_count == 0

print("\n=== Opposite subset check ===")
print(f"Model=1 total: {len(model_1_rows):,}")
print(f"Those also regulatory_according_to_ruleed 1: {true_subset_model_count:,}")
print(f"Model=1 but regulatory_according_to_rule=0: {false_subset_model_count:,}")
print(f"Are all model=1 rows also regulatory_according_to_rule=1? -> {is_subset_model}")

# Optional: show a few model=1 but regulatory_according_to_rule=0 sentences
if not is_subset_model and false_subset_model_count > 0:
    mismatches_model = model_1_rows[model_1_rows["regulatory_according_to_rule"] == 0]
    print("\nSample of mismatches (model=1, regulatory_according_to_rule=0):")
    for i, row in mismatches_model.head(5).iterrows():
        print(f"- {row['sent'][:120]}{'...' if len(row['sent']) > 120 else ''}")
    out_path = os.path.splitext(csv_path)[0] + "_model1_notregulatory_according_to_rule1.csv"
    mismatches_model.to_csv(out_path, index=False)
    print(f"[INFO] Saved all {false_subset_model_count:,} mismatched sentences to: {out_path}")

# --- Basic regulatory_according_to_rule statistics ---
print("\n=== regulatory_according_to_rule counts ===")
for col in ["regulatory_according_to_rule", "regulatory_according_to_inlegalbert"]:
    counts = df[col].value_counts().sort_index()
    zeros = counts.get(0, 0)
    ones = counts.get(1, 0)
    print(f"{col}: 0s = {zeros:,}, 1s = {ones:,}")

# --- Compare predictions ---
same_mask = df["regulatory_according_to_rule"] == df["regulatory_according_to_inlegalbert"]
agree = int(same_mask.sum())
disagree = len(df) - agree
agreement_ratio = agree / len(df) if len(df) else 0

print("\n=== Comparison ===")
print(f"Matching rows (agreement): {agree:,}")
print(f"Different rows: {disagree:,}")
print(f"Agreement ratio: {agreement_ratio:.3%}")

# --- Detailed metrics ---
y_true = df["regulatory_according_to_rule"]
y_pred = df["regulatory_according_to_inlegalbert"]
cm = confusion_matrix(y_true, y_pred)
acc = accuracy_score(y_true, y_pred)

print("\n=== Detailed metrics ===")
print("Confusion matrix [[TN FP]\n                   [FN TP]]:")
print(cm)
print(f"Accuracy: {acc:.3%}")

# --- Analyze sentences where both are 1 ---
matching_1 = df[(df["regulatory_according_to_rule"] == 1) & (df["regulatory_according_to_inlegalbert"] == 1)]
print(f"\n=== Sentences where both regulatory_according_to_rule & model = 1 (regulatory) ===")
print(f"Count: {len(matching_1):,}")

if len(matching_1) > 0:
    # Compute sentence length and word count
    matching_1 = matching_1.copy()
    matching_1["char_length"] = matching_1["sent"].apply(len)
    matching_1["word_count"] = matching_1["sent"].apply(lambda s: len(str(s).split()))

    # Summary stats
    print("\nCharacter length stats:")
    print(matching_1["char_length"].describe().to_string())

    print("\nWord count stats:")
    print(matching_1["word_count"].describe().to_string())

    # Preview first few
    print("\nSample matching regulatory sentences:")
    for i, row in matching_1.head(5).iterrows():
        print(f"- ({row['word_count']} words, {row['char_length']} chars) {row['sent'][:120]}{'...' if len(row['sent'])>120 else ''}")

    # Optionally save this subset for inspection
    out_path = os.path.splitext(csv_path)[0] + "_matching1_sentences.csv"
    matching_1.to_csv(out_path, index=False)
    print(f"\n[INFO] Saved matching sentences to: {out_path}")
else:
    print("No rows where both columns = 1.")
