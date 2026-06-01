# ============================================================
# Automatic ICD Category Prediction from Spanish Clinical Text
# ============================================================

import re
import unicodedata
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.base import clone
from IPython.display import display

warnings.filterwarnings("ignore")


# ============================================================
# 1. Load data
# ============================================================

TRAIN_PATH = "codification_data.csv"
ICD_PATH = "icd_d_p_pairs.csv"
LEADERBOARD_PATH = "leaderboard_data.csv"

train_df = pd.read_csv(TRAIN_PATH)
icd_df = pd.read_csv(ICD_PATH)
leaderboard_df = pd.read_csv(LEADERBOARD_PATH)

print("Training data:", train_df.shape)
print("ICD descriptions:", icd_df.shape)
print("Leaderboard data:", leaderboard_df.shape)

print("\nTrain columns:", train_df.columns.tolist())
print("ICD columns:", icd_df.columns.tolist())
print("Leaderboard columns:", leaderboard_df.columns.tolist())

display(train_df.head())
display(icd_df.head())
display(leaderboard_df.head())


# ============================================================
# 2. Create target label: y_category
# ============================================================

# Official task: y_category = first character of ICD Code.
# Example: I420 -> i, 3E03329 -> 3

train_df["Code"] = train_df["Code"].astype(str)
train_df["Literal"] = train_df["Literal"].astype(str)

train_df["y_category"] = train_df["Code"].str[0].str.lower()

icd_df["Code"] = icd_df["Code"].astype(str)
icd_df["Description"] = icd_df["Description"].astype(str)
icd_df["y_category"] = icd_df["Code"].str[0].str.lower()

leaderboard_df["Literal"] = leaderboard_df["Literal"].astype(str)

labels = sorted(train_df["y_category"].unique())

print("Number of classes:", len(labels))
print("Classes:", labels)


# ============================================================
# 3. Text normalization
# ============================================================

def normalize_text(text):
    """
    Normalizes short clinical literals:
    - converts to string
    - lowercases
    - removes accents
    - removes punctuation
    - collapses spaces
    """
    if pd.isna(text):
        return ""

    text = str(text).strip().lower()

    # Remove accents: "miocardiopatía" -> "miocardiopatia"
    text = unicodedata.normalize("NFKD", text)
    text = "".join(ch for ch in text if not unicodedata.combining(ch))

    # Replace punctuation/symbols with spaces
    text = re.sub(r"[^a-z0-9\s]", " ", text)

    # Collapse multiple spaces
    text = re.sub(r"\s+", " ", text).strip()

    return text


train_df["norm_literal"] = train_df["Literal"].apply(normalize_text)
leaderboard_df["norm_literal"] = leaderboard_df["Literal"].apply(normalize_text)
icd_df["norm_description"] = icd_df["Description"].apply(normalize_text)

display(train_df[["Literal", "norm_literal", "Code", "y_category"]].head(10))


# ============================================================
# 4. Dataset analysis
# ============================================================

print("\nDataset size")
print("Train examples:", len(train_df))
print("Leaderboard examples:", len(leaderboard_df))
print("ICD descriptions:", len(icd_df))

print("\nClass distribution:")
class_counts = train_df["y_category"].value_counts().sort_index()
display(class_counts.to_frame("count"))

plt.figure(figsize=(12, 4))
class_counts.plot(kind="bar")
plt.title("Class distribution of y_category")
plt.xlabel("ICD category")
plt.ylabel("Number of examples")
plt.tight_layout()
plt.show()


# Shortest and longest literals
train_df["literal_len"] = train_df["Literal"].str.len()
train_df["n_words"] = train_df["Literal"].str.split().apply(len)

print("\nLiteral length statistics:")
display(train_df[["literal_len", "n_words"]].describe())

print("\nExamples of very short literals:")
display(train_df.sort_values("literal_len")[["Literal", "Code", "y_category"]].head(20))

print("\nExamples of longer literals:")
display(train_df.sort_values("literal_len", ascending=False)[["Literal", "Code", "y_category"]].head(20))


# Duplicated normalized literals
dup_stats = (
    train_df.groupby("norm_literal")
    .agg(
        n_examples=("Literal", "size"),
        n_categories=("y_category", "nunique"),
        categories=("y_category", lambda x: ", ".join(sorted(set(x)))),
        example_literal=("Literal", "first")
    )
    .reset_index()
)

duplicates = dup_stats[dup_stats["n_examples"] > 1].sort_values(
    ["n_categories", "n_examples"], ascending=False
)

conflicts = duplicates[duplicates["n_categories"] > 1]

print("\nNumber of duplicated normalized literals:", len(duplicates))
print("Number of duplicated literals with conflicting categories:", len(conflicts))

print("\nExamples of conflicting literals:")
display(conflicts.head(20))


# ============================================================
# 5. Train / validation split
# ============================================================

train_split, val_split = train_test_split(
    train_df,
    test_size=0.2,
    random_state=42,
    stratify=train_df["y_category"]
)

print("Train split:", train_split.shape)
print("Validation split:", val_split.shape)

X_train = train_split["Literal"]
y_train = train_split["y_category"]

X_val = val_split["Literal"]
y_val = val_split["y_category"]


# ============================================================
# 6. Evaluation helper
# ============================================================

results = []

def evaluate_predictions(name, y_true, y_pred, store=True):
    acc = accuracy_score(y_true, y_pred)
    macro = f1_score(y_true, y_pred, average="macro", zero_division=0)
    weighted = f1_score(y_true, y_pred, average="weighted", zero_division=0)

    print("\n" + "=" * 70)
    print(name)
    print("=" * 70)
    print(f"Accuracy:    {acc:.4f}")
    print(f"Macro-F1:    {macro:.4f}")
    print(f"Weighted-F1: {weighted:.4f}")
    print("\nClassification report:")
    print(classification_report(y_true, y_pred, labels=labels, zero_division=0))

    if store:
        results.append({
            "method": name,
            "accuracy": acc,
            "macro_f1": macro,
            "weighted_f1": weighted
        })

    return acc, macro, weighted


# ============================================================
# 7. Baseline 0: Majority class
# ============================================================

majority_class = y_train.value_counts().idxmax()
majority_pred = np.array([majority_class] * len(y_val))

evaluate_predictions("Majority baseline", y_val, majority_pred)


# ============================================================
# 8. Method 1: Lexical baseline
#    Normalized exact matching against training literals
# ============================================================

def most_common_label(series):
    return series.value_counts().idxmax()

literal_to_label = (
    train_split.groupby("norm_literal")["y_category"]
    .agg(most_common_label)
    .to_dict()
)

def predict_exact_literal_match(literals, mapping, fallback):
    preds = []
    for text in literals:
        norm = normalize_text(text)
        pred = mapping.get(norm, fallback)
        preds.append(pred)
    return np.array(preds)

lexical_pred = predict_exact_literal_match(
    X_val,
    literal_to_label,
    fallback=majority_class
)

evaluate_predictions("Lexical baseline: exact normalized literal match", y_val, lexical_pred)


# ============================================================
# 9. Extra lexical baseline:
#    Exact matching against ICD descriptions
# ============================================================

icd_desc_to_label = (
    icd_df.groupby("norm_description")["y_category"]
    .agg(most_common_label)
    .to_dict()
)

icd_exact_pred = predict_exact_literal_match(
    X_val,
    icd_desc_to_label,
    fallback=majority_class
)

evaluate_predictions("Lexical baseline: exact ICD description match", y_val, icd_exact_pred)


# ============================================================
# 10. Method 2: Semantic retrieval with TF-IDF + cosine similarity
# ============================================================

def build_icd_retriever(
    analyzer="char_wb",
    ngram_range=(3, 5),
    max_features=200_000
):
    """
    Fits a TF-IDF vectorizer on ICD descriptions.
    Returns vectorizer, ICD matrix, and ICD label array.
    """
    vectorizer = TfidfVectorizer(
        analyzer=analyzer,
        ngram_range=ngram_range,
        lowercase=False,
        preprocessor=None,
        min_df=1,
        max_features=max_features,
        sublinear_tf=True,
        norm="l2",
        dtype=np.float32
    )

    icd_texts = icd_df["norm_description"].fillna("").tolist()
    X_icd = vectorizer.fit_transform(icd_texts)
    icd_labels = icd_df["y_category"].values

    return vectorizer, X_icd, icd_labels


def retrieval_predict(literals, vectorizer, X_icd, icd_labels, batch_size=128):
    """
    For each literal:
    - transform literal with the same TF-IDF vectorizer
    - compute cosine similarity with ICD descriptions
    - return the category of the nearest ICD description
    """
    normalized_literals = [normalize_text(x) for x in literals]
    preds = []

    for start in range(0, len(normalized_literals), batch_size):
        batch_texts = normalized_literals[start:start + batch_size]
        X_batch = vectorizer.transform(batch_texts)

        # Because TF-IDF vectors are L2-normalized, dot product = cosine similarity.
        sims = X_batch @ X_icd.T

        best_indices = np.asarray(sims.argmax(axis=1)).reshape(-1)
        batch_preds = icd_labels[best_indices]

        preds.extend(batch_preds)

    return np.array(preds)


# ---- Retrieval variant A: word n-grams ----

print("\nFitting ICD retrieval vectorizer: word n-grams...")
word_retriever, X_icd_word, icd_labels = build_icd_retriever(
    analyzer="word",
    ngram_range=(1, 2),
    max_features=100_000
)

retrieval_word_pred = retrieval_predict(
    X_val,
    word_retriever,
    X_icd_word,
    icd_labels,
    batch_size=128
)

evaluate_predictions("Semantic retrieval: TF-IDF word n-grams", y_val, retrieval_word_pred)


# ---- Retrieval variant B: character n-grams ----

print("\nFitting ICD retrieval vectorizer: character n-grams...")
char_retriever, X_icd_char, icd_labels = build_icd_retriever(
    analyzer="char_wb",
    ngram_range=(3, 5),
    max_features=200_000
)

retrieval_char_pred = retrieval_predict(
    X_val,
    char_retriever,
    X_icd_char,
    icd_labels,
    batch_size=128
)

evaluate_predictions("Semantic retrieval: TF-IDF character n-grams", y_val, retrieval_char_pred)


# ============================================================
# 11. Method 3: Supervised classification
# ============================================================

supervised_models = {}

# Logistic Regression + word TF-IDF
supervised_models["Logistic Regression: word TF-IDF"] = Pipeline([
    ("tfidf", TfidfVectorizer(
        preprocessor=normalize_text,
        lowercase=False,
        analyzer="word",
        ngram_range=(1, 2),
        min_df=1,
        max_features=100_000,
        sublinear_tf=True
    )),
    ("clf", LogisticRegression(
        max_iter=2000,
        solver="liblinear",
        class_weight="balanced",
        random_state=42
    ))
])

# Logistic Regression + character TF-IDF
supervised_models["Logistic Regression: char TF-IDF"] = Pipeline([
    ("tfidf", TfidfVectorizer(
        preprocessor=normalize_text,
        lowercase=False,
        analyzer="char_wb",
        ngram_range=(3, 5),
        min_df=1,
        max_features=200_000,
        sublinear_tf=True
    )),
    ("clf", LogisticRegression(
        max_iter=2000,
        solver="liblinear",
        class_weight="balanced",
        random_state=42
    ))
])

# Linear SVM + word TF-IDF
supervised_models["Linear SVM: word TF-IDF"] = Pipeline([
    ("tfidf", TfidfVectorizer(
        preprocessor=normalize_text,
        lowercase=False,
        analyzer="word",
        ngram_range=(1, 2),
        min_df=1,
        max_features=100_000,
        sublinear_tf=True
    )),
    ("clf", LinearSVC(
        C=1.0,
        class_weight="balanced",
        random_state=42
    ))
])

# Linear SVM + character TF-IDF
supervised_models["Linear SVM: char TF-IDF"] = Pipeline([
    ("tfidf", TfidfVectorizer(
        preprocessor=normalize_text,
        lowercase=False,
        analyzer="char_wb",
        ngram_range=(3, 5),
        min_df=1,
        max_features=200_000,
        sublinear_tf=True
    )),
    ("clf", LinearSVC(
        C=1.0,
        class_weight="balanced",
        random_state=42
    ))
])


fitted_supervised_models = {}
supervised_scores = {}

for name, model in supervised_models.items():
    print("\nTraining:", name)
    model.fit(X_train, y_train)

    pred = model.predict(X_val)

    acc, macro, weighted = evaluate_predictions(name, y_val, pred)

    fitted_supervised_models[name] = model
    supervised_scores[name] = {
        "accuracy": acc,
        "macro_f1": macro,
        "weighted_f1": weighted
    }


# ============================================================
# 12. Compare all results
# ============================================================

results_df = pd.DataFrame(results).sort_values(
    by=["macro_f1", "accuracy"],
    ascending=False
)

print("\nValidation results:")
display(results_df)

results_df.to_csv("validation_results.csv", index=False)


# ============================================================
# 13. Pick best supervised model
# ============================================================

best_supervised_name = max(
    supervised_scores,
    key=lambda name: supervised_scores[name]["macro_f1"]
)

best_model = fitted_supervised_models[best_supervised_name]

print("\nBest supervised model:", best_supervised_name)
print(supervised_scores[best_supervised_name])


# ============================================================
# 14. Confusion matrix for best model
# ============================================================

best_val_pred = best_model.predict(X_val)

cm = confusion_matrix(y_val, best_val_pred, labels=labels)

plt.figure(figsize=(12, 12))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
disp.plot(
    xticks_rotation=90,
    values_format="d",
    cmap=None
)
plt.title(f"Confusion matrix: {best_supervised_name}")
plt.tight_layout()
plt.show()


# ============================================================
# 15. Error analysis
# ============================================================

error_df = val_split.copy()
error_df["prediction"] = best_val_pred
error_df["correct"] = error_df["prediction"] == error_df["y_category"]

errors_only = error_df[~error_df["correct"]].copy()

print("\nNumber of validation errors:", len(errors_only))
display(errors_only[["Literal", "Code", "y_category", "prediction"]].head(50))

errors_only.to_csv("validation_errors_best_model.csv", index=False)


# Most common confusions
confusions = (
    errors_only.groupby(["y_category", "prediction"])
    .size()
    .reset_index(name="count")
    .sort_values("count", ascending=False)
)

print("\nMost common confusions:")
display(confusions.head(30))

confusions.to_csv("common_confusions.csv", index=False)


# Per-class performance
report_dict = classification_report(
    y_val,
    best_val_pred,
    labels=labels,
    output_dict=True,
    zero_division=0
)

per_class_df = (
    pd.DataFrame(report_dict)
    .transpose()
    .reset_index()
    .rename(columns={"index": "class"})
)

per_class_df = per_class_df[per_class_df["class"].isin(labels)]
per_class_df = per_class_df.sort_values("f1-score")

print("\nWorst classes by F1:")
display(per_class_df.head(15))

print("\nBest classes by F1:")
display(per_class_df.tail(15))

per_class_df.to_csv("per_class_report_best_model.csv", index=False)


# ============================================================
# 16. Train best supervised model on the full training data
# ============================================================

print("\nRetraining best supervised model on full training data...")

final_model = clone(supervised_models[best_supervised_name])
final_model.fit(train_df["Literal"], train_df["y_category"])

leaderboard_pred = final_model.predict(leaderboard_df["Literal"])

submission = pd.DataFrame({
    "id": leaderboard_df["id"],
    "y_category": leaderboard_pred
})

submission.to_csv("submission_best_supervised.csv", index=False)

print("\nSaved: submission_best_supervised.csv")
display(submission.head(20))


# ============================================================
# 17. Optional: also create retrieval submission
# ============================================================

# This creates a submission using the character n-gram retrieval method.
# It may be worse than the supervised model, but it is useful for comparison.

print("\nCreating retrieval-based submission...")

retrieval_leaderboard_pred = retrieval_predict(
    leaderboard_df["Literal"],
    char_retriever,
    X_icd_char,
    icd_labels,
    batch_size=128
)

retrieval_submission = pd.DataFrame({
    "id": leaderboard_df["id"],
    "y_category": retrieval_leaderboard_pred
})

retrieval_submission.to_csv("submission_retrieval_char.csv", index=False)

print("Saved: submission_retrieval_char.csv")
display(retrieval_submission.head(20))


# ============================================================
# 18. Optional: also create lexical submission
# ============================================================

print("\nCreating lexical exact-match submission...")

# Rebuild lexical mapping using the full training data
full_literal_to_label = (
    train_df.groupby("norm_literal")["y_category"]
    .agg(most_common_label)
    .to_dict()
)

full_majority_class = train_df["y_category"].value_counts().idxmax()

lexical_leaderboard_pred = predict_exact_literal_match(
    leaderboard_df["Literal"],
    full_literal_to_label,
    fallback=full_majority_class
)

lexical_submission = pd.DataFrame({
    "id": leaderboard_df["id"],
    "y_category": lexical_leaderboard_pred
})

lexical_submission.to_csv("submission_lexical_exact_match.csv", index=False)

print("Saved: submission_lexical_exact_match.csv")
display(lexical_submission.head(20))