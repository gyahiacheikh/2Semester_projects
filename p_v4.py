# ============================================================
# NLP-I Task: ICD-10 Codification
# RoBERTa Biomedical Clinical Spanish
# Version using icd_d_p_pairs.csv
# ============================================================

import re
import copy
import random
import warnings
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

from transformers import AutoTokenizer, AutoModel

warnings.filterwarnings("ignore")


# ============================================================
# 1. Configuration
# ============================================================

try:
    BASE_DIR = Path(__file__).resolve().parent
except NameError:
    BASE_DIR = Path.cwd().resolve()

TRAIN_PATH = BASE_DIR / "codification_data.csv"
LEADERBOARD_PATH = BASE_DIR / "leaderboard_data.csv"
PAIRS_PATH = BASE_DIR / "icd_d_p_pairs.csv"

OUTPUT_DIR = BASE_DIR / "outputs_roberta_icd"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_NAME = "PlanTL-GOB-ES/roberta-base-biomedical-clinical-es"

SEED = 42
MAX_EPOCHS = 50
PATIENCE = 10
BATCH_SIZE = 128
LEARNING_RATE = 2e-5
WEIGHT_DECAY = 0.01
DROPOUT = 0.1
MAX_LENGTH = 64

POOLING = "mean"
NUM_WORKERS = 0


# ============================================================
# 2. Reproducibility
# ============================================================

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


set_seed(SEED)


# ============================================================
# 3. Text preprocessing
# ============================================================

def preprocess_text(text):
    if pd.isna(text):
        return ""

    text = str(text)
    text = text.strip()
    text = re.sub(r"\s+", " ", text)

    return text


def normalize_for_lookup(text):
    """
    Stronger normalization only for exact/near-exact lookup.
    This is NOT used as RoBERTa input.
    """
    text = preprocess_text(text)
    text = text.lower()

    text = unicodedata.normalize("NFD", text)
    text = "".join(
        ch for ch in text
        if unicodedata.category(ch) != "Mn"
    )

    text = re.sub(r"[^a-z0-9]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    return text


# ============================================================
# 4. Utility functions
# ============================================================

def check_file_exists(path, name):
    if not path.exists():
        raise FileNotFoundError(
            f"{name} not found at: {path.resolve()}"
        )


def save_csv_checked(df, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8")

    if not path.exists():
        raise RuntimeError(f"File was not saved correctly: {path.resolve()}")


def pick_col(df, candidates, required=True):
    for col in candidates:
        if col in df.columns:
            return col

    if required:
        raise ValueError(
            f"None of these columns were found: {candidates}. "
            f"Available columns: {list(df.columns)}"
        )

    return None


def build_majority_lookup(df, key_function):
    """
    Builds:
        processed_literal -> most frequent y_category

    If the same literal appears with several categories, the most common one wins.
    """
    temp = df.copy()
    temp["_lookup_key"] = temp["Literal"].apply(key_function)
    temp = temp[temp["_lookup_key"].str.len() > 0].copy()

    counts = (
        temp
        .groupby(["_lookup_key", "y_category"], as_index=False)
        .size()
        .rename(columns={"size": "count"})
    )

    counts = counts.sort_values(
        by=["_lookup_key", "count", "y_category"],
        ascending=[True, False, True]
    )

    best = counts.drop_duplicates(subset=["_lookup_key"], keep="first")

    return dict(zip(best["_lookup_key"], best["y_category"]))


def apply_lookup_fallback(texts, model_pred_labels, exact_lookup, normalized_lookup):
    """
    Final prediction rule:
    1. Exact literal match.
    2. Normalized literal match.
    3. RoBERTa prediction.
    """
    final_preds = []

    for text, model_pred in zip(texts, model_pred_labels):
        exact_key = preprocess_text(text)
        normalized_key = normalize_for_lookup(text)

        if exact_key in exact_lookup:
            final_preds.append(exact_lookup[exact_key])

        elif normalized_key in normalized_lookup:
            final_preds.append(normalized_lookup[normalized_key])

        else:
            final_preds.append(model_pred)

    return np.array(final_preds)


# ============================================================
# 5. Load data
# ============================================================

check_file_exists(TRAIN_PATH, "codification_data.csv")
check_file_exists(LEADERBOARD_PATH, "leaderboard_data.csv")
check_file_exists(PAIRS_PATH, "icd_d_p_pairs.csv")

train_df = pd.read_csv(
    TRAIN_PATH,
    dtype={
        "Code": str,
        "Literal": str
    }
)

leaderboard_df = pd.read_csv(
    LEADERBOARD_PATH,
    dtype={
        "Literal": str
    }
)

pairs_df = pd.read_csv(
    PAIRS_PATH,
    dtype=str
)

required_train_cols = {"Code", "Literal"}
required_leaderboard_cols = {"id", "Literal"}

missing_train = required_train_cols - set(train_df.columns)
missing_leaderboard = required_leaderboard_cols - set(leaderboard_df.columns)

if missing_train:
    raise ValueError(f"Missing required training columns: {missing_train}")

if missing_leaderboard:
    raise ValueError(f"Missing required leaderboard columns: {missing_leaderboard}")

pairs_code_col = pick_col(
    pairs_df,
    ["code", "Code", "ICD_CODE", "ICD Code", "icd_code"]
)

pairs_description_col = pick_col(
    pairs_df,
    ["Description", "description", "DESCRIPTION"]
)

leaderboard_df["Literal_original"] = leaderboard_df["Literal"].fillna("").astype(str)

# ------------------------------
# Clean supervised codification data
# ------------------------------

train_df["Code"] = train_df["Code"].fillna("").astype(str).str.strip()
train_df["Literal"] = train_df["Literal"].apply(preprocess_text)

train_df = train_df[
    (train_df["Code"].str.len() > 0)
    & (train_df["Literal"].str.len() > 0)
].copy()

train_df["y_category"] = train_df["Code"].astype(str).str[0]
train_df["source"] = "codification_data"

# ------------------------------
# Convert official ICD pairs into extra training examples
# ------------------------------

official_df = pd.DataFrame({
    "Code": pairs_df[pairs_code_col].fillna("").astype(str).str.strip(),
    "Literal": pairs_df[pairs_description_col].apply(preprocess_text)
})

official_df = official_df[
    (official_df["Code"].str.len() > 0)
    & (official_df["Literal"].str.len() > 0)
].copy()

official_df["y_category"] = official_df["Code"].astype(str).str[0]
official_df["source"] = "icd_d_p_pairs"

# ------------------------------
# Clean leaderboard data
# ------------------------------

leaderboard_df["Literal"] = leaderboard_df["Literal"].apply(preprocess_text)

# ------------------------------
# Labels from both supervised data and official ICD pairs
# ------------------------------

all_label_values = pd.concat(
    [
        train_df["y_category"],
        official_df["y_category"]
    ],
    axis=0
)

labels = sorted(all_label_values.unique())
label2id = {label: idx for idx, label in enumerate(labels)}
id2label = {idx: label for label, idx in label2id.items()}

train_df["label_id"] = train_df["y_category"].map(label2id)
official_df["label_id"] = official_df["y_category"].map(label2id)


# ============================================================
# 6. Train / validation split
# ============================================================

class_counts = train_df["label_id"].value_counts()

if class_counts.min() >= 2:
    stratify_values = train_df["label_id"]
else:
    stratify_values = None

train_split, val_split = train_test_split(
    train_df,
    test_size=0.2,
    random_state=SEED,
    stratify=stratify_values
)

# Validation stays real: only codification_data.csv examples.
# Training is augmented with official ICD descriptions.
train_augmented = pd.concat(
    [
        train_split,
        official_df
    ],
    axis=0,
    ignore_index=True
)

train_augmented = train_augmented.sample(
    frac=1,
    random_state=SEED
).reset_index(drop=True)

# Lookups for validation:
# Built from train_split + official descriptions, NOT from val_split.
validation_lookup_df = pd.concat(
    [
        train_split[["Literal", "y_category"]],
        official_df[["Literal", "y_category"]]
    ],
    axis=0,
    ignore_index=True
)

validation_exact_lookup = build_majority_lookup(
    validation_lookup_df,
    preprocess_text
)

validation_normalized_lookup = build_majority_lookup(
    validation_lookup_df,
    normalize_for_lookup
)

# Lookups for final leaderboard:
# For the final submission, all labelled data can be used.
final_lookup_df = pd.concat(
    [
        train_df[["Literal", "y_category"]],
        official_df[["Literal", "y_category"]]
    ],
    axis=0,
    ignore_index=True
)

final_exact_lookup = build_majority_lookup(
    final_lookup_df,
    preprocess_text
)

final_normalized_lookup = build_majority_lookup(
    final_lookup_df,
    normalize_for_lookup
)


# ============================================================
# 7. Dataset
# ============================================================

class ICDDataset(Dataset):
    def __init__(self, texts, tokenizer, max_length, labels=None):
        self.texts = [str(text) for text in texts]
        self.labels = None if labels is None else list(labels)

        self.encodings = tokenizer(
            self.texts,
            padding="max_length",
            truncation=True,
            max_length=max_length,
            return_tensors="pt"
        )

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        item = {
            "input_ids": self.encodings["input_ids"][idx],
            "attention_mask": self.encodings["attention_mask"][idx]
        }

        if self.labels is not None:
            item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)

        return item


# ============================================================
# 8. Model
# ============================================================

class ICDRobertaClassifier(nn.Module):
    def __init__(self, model_name, num_classes, pooling="mean", dropout=0.1):
        super().__init__()

        if pooling not in {"cls", "mean"}:
            raise ValueError("pooling must be 'cls' or 'mean'")

        self.pooling = pooling
        self.backbone = AutoModel.from_pretrained(model_name)

        hidden_size = self.backbone.config.hidden_size

        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size, num_classes)

    def forward(self, input_ids, attention_mask):
        outputs = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        hidden_states = outputs.last_hidden_state

        if self.pooling == "cls":
            features = hidden_states[:, 0, :]
        else:
            mask = attention_mask.unsqueeze(-1).float()
            summed = (hidden_states * mask).sum(dim=1)
            counts = mask.sum(dim=1).clamp(min=1e-9)
            features = summed / counts

        features = self.dropout(features)
        logits = self.classifier(features)

        return logits


# ============================================================
# 9. Training and evaluation functions
# ============================================================

def train_one_epoch(model, dataloader, optimizer, criterion, device):
    model.train()

    total_loss = 0.0
    total_examples = 0

    for batch in dataloader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels_batch = batch["labels"].to(device)

        optimizer.zero_grad(set_to_none=True)

        logits = model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        loss = criterion(logits, labels_batch)

        loss.backward()
        optimizer.step()

        batch_size = labels_batch.size(0)
        total_loss += loss.item() * batch_size
        total_examples += batch_size

    return total_loss / total_examples


def evaluate(model, dataloader, criterion, device):
    model.eval()

    total_loss = 0.0
    total_examples = 0

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels_batch = batch["labels"].to(device)

            logits = model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )

            loss = criterion(logits, labels_batch)

            preds = torch.argmax(logits, dim=1)

            batch_size = labels_batch.size(0)
            total_loss += loss.item() * batch_size
            total_examples += batch_size

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels_batch.cpu().numpy())

    val_loss = total_loss / total_examples
    val_acc = accuracy_score(all_labels, all_preds)

    return val_loss, val_acc, np.array(all_labels), np.array(all_preds)


def predict(model, dataloader, device):
    model.eval()

    all_preds = []

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            logits = model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )

            preds = torch.argmax(logits, dim=1)
            all_preds.extend(preds.cpu().numpy())

    return np.array(all_preds)


# ============================================================
# 10. Prepare tokenizer and dataloaders
# ============================================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

train_dataset = ICDDataset(
    texts=train_augmented["Literal"],
    labels=train_augmented["label_id"],
    tokenizer=tokenizer,
    max_length=MAX_LENGTH
)

val_dataset = ICDDataset(
    texts=val_split["Literal"],
    labels=val_split["label_id"],
    tokenizer=tokenizer,
    max_length=MAX_LENGTH
)

leaderboard_dataset = ICDDataset(
    texts=leaderboard_df["Literal"],
    labels=None,
    tokenizer=tokenizer,
    max_length=MAX_LENGTH
)

pin_memory = device.type == "cuda"

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=NUM_WORKERS,
    pin_memory=pin_memory
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=pin_memory
)

leaderboard_loader = DataLoader(
    leaderboard_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=pin_memory
)


# ============================================================
# 11. Initialize model
# ============================================================

model = ICDRobertaClassifier(
    model_name=MODEL_NAME,
    num_classes=len(labels),
    pooling=POOLING,
    dropout=DROPOUT
)

model = model.to(device)

criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY
)


# ============================================================
# 12. Fine-tuning loop with early stopping
# ============================================================

best_val_acc = -1.0
best_epoch = 0
epochs_without_improvement = 0
best_state_dict = None

history = []

checkpoint_path = OUTPUT_DIR / f"best_roberta_{POOLING}_with_pairs.pt"

for epoch in range(1, MAX_EPOCHS + 1):
    train_loss = train_one_epoch(
        model=model,
        dataloader=train_loader,
        optimizer=optimizer,
        criterion=criterion,
        device=device
    )

    val_loss, raw_val_acc, val_true_ids, val_pred_ids = evaluate(
        model=model,
        dataloader=val_loader,
        criterion=criterion,
        device=device
    )

    val_true_labels = np.array([
        id2label[int(idx)] for idx in val_true_ids
    ])

    raw_val_pred_labels = np.array([
        id2label[int(idx)] for idx in val_pred_ids
    ])

    final_val_pred_labels = apply_lookup_fallback(
        texts=val_split["Literal"].values,
        model_pred_labels=raw_val_pred_labels,
        exact_lookup=validation_exact_lookup,
        normalized_lookup=validation_normalized_lookup
    )

    val_acc = accuracy_score(val_true_labels, final_val_pred_labels)

    history.append({
        "epoch": epoch,
        "train_loss": train_loss,
        "val_loss": val_loss,
        "raw_val_accuracy": raw_val_acc,
        "val_accuracy": val_acc
    })

    print(
        f"Epoch {epoch:02d} | "
        f"train_loss={train_loss:.4f} | "
        f"val_loss={val_loss:.4f} | "
        f"val_acc={val_acc:.4f}"
    )

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        best_epoch = epoch
        epochs_without_improvement = 0
        best_state_dict = copy.deepcopy(model.state_dict())

        torch.save(
            {
                "model_state_dict": best_state_dict,
                "model_name": MODEL_NAME,
                "pooling": POOLING,
                "labels": labels,
                "label2id": label2id,
                "id2label": id2label,
                "max_length": MAX_LENGTH,
                "val_accuracy": best_val_acc,
                "epoch": best_epoch
            },
            checkpoint_path
        )

        print(f"Saved best model: {checkpoint_path.resolve()}")
        #print(f"Checkpoint exists: {checkpoint_path.exists()}")

    else:
        epochs_without_improvement += 1
        print(f"No improvement for {epochs_without_improvement} epoch(s).")

    if epochs_without_improvement >= PATIENCE:
        print("Early stopping triggered.")
        break


history_df = pd.DataFrame(history)

save_csv_checked(
    history_df,
    OUTPUT_DIR / f"training_history_{POOLING}_with_pairs.csv"
)

print("Best epoch:", best_epoch)
print("Best validation accuracy:", best_val_acc)


# ============================================================
# 13. Load best model and final validation report
# ============================================================

if not checkpoint_path.exists():
    raise FileNotFoundError(
        f"No checkpoint found at {checkpoint_path.resolve()}."
    )

try:
    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False
    )
except TypeError:
    checkpoint = torch.load(
        checkpoint_path,
        map_location=device
    )

model.load_state_dict(checkpoint["model_state_dict"])
model = model.to(device)

val_loss, raw_val_acc, val_true_ids, val_pred_ids = evaluate(
    model=model,
    dataloader=val_loader,
    criterion=criterion,
    device=device
)

val_true_labels = np.array([
    id2label[int(idx)] for idx in val_true_ids
])

raw_val_pred_labels = np.array([
    id2label[int(idx)] for idx in val_pred_ids
])

final_val_pred_labels = apply_lookup_fallback(
    texts=val_split["Literal"].values,
    model_pred_labels=raw_val_pred_labels,
    exact_lookup=validation_exact_lookup,
    normalized_lookup=validation_normalized_lookup
)

print("\nClassification report:")
print(
    classification_report(
        val_true_labels,
        final_val_pred_labels,
        labels=labels,
        zero_division=0
    )
)

val_predictions_df = pd.DataFrame({
    "Literal": val_split["Literal"].values,
    "Code": val_split["Code"].values,
    "true_y_category": val_true_labels,
    "roberta_prediction": raw_val_pred_labels,
    "final_prediction": final_val_pred_labels
})

val_predictions_df["is_correct"] = (
    val_predictions_df["true_y_category"]
    == val_predictions_df["final_prediction"]
)

save_csv_checked(
    val_predictions_df,
    OUTPUT_DIR / f"validation_predictions_{POOLING}_with_pairs.csv"
)


# ============================================================
# 14. Predict leaderboard data
# ============================================================

leaderboard_pred_ids = predict(
    model=model,
    dataloader=leaderboard_loader,
    device=device
)

leaderboard_pred_labels = np.array([
    id2label[int(idx)] for idx in leaderboard_pred_ids
])

final_leaderboard_pred_labels = apply_lookup_fallback(
    texts=leaderboard_df["Literal"].values,
    model_pred_labels=leaderboard_pred_labels,
    exact_lookup=final_exact_lookup,
    normalized_lookup=final_normalized_lookup
)

submission = pd.DataFrame({
    "id": leaderboard_df["id"],
    "Literal": leaderboard_df["Literal_original"],
    "y_category": final_leaderboard_pred_labels
})

submission["y_category"] = (
    submission["y_category"]
    .fillna("null")
    .astype(str)
    .replace("", "null")
)

expected_columns = ["id", "Literal", "y_category"]

if list(submission.columns) != expected_columns:
    raise ValueError(
        f"Wrong submission columns. Expected {expected_columns}, "
        f"got {list(submission.columns)}"
    )

if submission["y_category"].isna().any():
    raise ValueError("Submission contains missing y_category values.")

if (submission["y_category"].astype(str).str.len() == 0).any():
    raise ValueError("Submission contains empty y_category values.")

if len(submission) != len(leaderboard_df):
    raise ValueError(
        f"Submission row count mismatch: submission={len(submission)}, "
        f"leaderboard={len(leaderboard_df)}"
    )

submission_path = OUTPUT_DIR / f"submission_roberta_{POOLING}_with_pairs.csv"

save_csv_checked(submission, submission_path)

print("\nSaved submission:")
print(submission_path.resolve())