# ============================================================
# NLP-I Task: ICD-10 Codification
# Deep Learning Baseline with RoBERTa Biomedical Clinical Spanish
# ============================================================

import re
import copy
import random
import warnings
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

BASE_DIR = Path(__file__).resolve().parent
TRAIN_PATH = BASE_DIR / "codification_data.csv"

LEADERBOARD_PATH = BASE_DIR / "leaderboard_data.csv"

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

POOLING = "mean"       # Choose "cls" or "mean"
NUM_WORKERS = 0        # Keep 0 on Windows


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
# 3. Light text preprocessing
# ============================================================

def preprocess_text(text):
    """
    Required light preprocessing:
    - convert to string
    - strip spaces
    - collapse multiple spaces

    Important:
    - do not lowercase
    - do not remove accents
    - do not remove punctuation
    """
    text = str(text)
    text = text.strip()
    text = re.sub(r"\s+", " ", text)
    return text


# ============================================================
# 4. Load data
# ============================================================

train_df = pd.read_csv(TRAIN_PATH)
leaderboard_df = pd.read_csv(LEADERBOARD_PATH)

required_train_cols = {"Code", "Literal"}
required_leaderboard_cols = {"id", "Literal"}

missing_train = required_train_cols - set(train_df.columns)
missing_leaderboard = required_leaderboard_cols - set(leaderboard_df.columns)

if missing_train:
    raise ValueError(f"Missing required training columns: {missing_train}")

if missing_leaderboard:
    raise ValueError(f"Missing required leaderboard columns: {missing_leaderboard}")

train_df["Code"] = train_df["Code"].astype(str)
train_df["Literal"] = train_df["Literal"].apply(preprocess_text)
leaderboard_df["Literal"] = leaderboard_df["Literal"].apply(preprocess_text)

train_df["y_category"] = train_df["Code"].astype(str).str[0].str.lower()

labels = sorted(train_df["y_category"].unique())
label2id = {label: idx for idx, label in enumerate(labels)}
id2label = {idx: label for label, idx in label2id.items()}

train_df["label_id"] = train_df["y_category"].map(label2id)

print("Training data:", train_df.shape)
print("Leaderboard data:", leaderboard_df.shape)
print("Number of classes:", len(labels))
print("Classes:", labels)


# ============================================================
# 5. Train / validation split
# ============================================================

train_split, val_split = train_test_split(
    train_df,
    test_size=0.2,
    random_state=SEED,
    stratify=train_df["label_id"]
)

print("Train split:", train_split.shape)
print("Validation split:", val_split.shape)


# ============================================================
# 6. Dataset
# ============================================================

class ICDDataset(Dataset):
    def __init__(self, texts, tokenizer, max_length, labels=None):
        self.texts = list(texts)
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
# 7. Model
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
# 8. Training and evaluation functions
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
# 9. Prepare tokenizer and dataloaders
# ============================================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

train_dataset = ICDDataset(
    texts=train_split["Literal"],
    labels=train_split["label_id"],
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

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=NUM_WORKERS
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS
)

leaderboard_loader = DataLoader(
    leaderboard_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS
)


# ============================================================
# 10. Initialize model
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
# 11. Fine-tuning loop with early stopping
# ============================================================

best_val_acc = 0.0
best_epoch = 0
epochs_without_improvement = 0
best_state_dict = None

history = []

checkpoint_path = OUTPUT_DIR / f"best_roberta_{POOLING}.pt"

for epoch in range(1, MAX_EPOCHS + 1):
    train_loss = train_one_epoch(
        model=model,
        dataloader=train_loader,
        optimizer=optimizer,
        criterion=criterion,
        device=device
    )

    val_loss, val_acc, val_true_ids, val_pred_ids = evaluate(
        model=model,
        dataloader=val_loader,
        criterion=criterion,
        device=device
    )

    history.append({
        "epoch": epoch,
        "train_loss": train_loss,
        "val_loss": val_loss,
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

        print(f"Saved best model: {checkpoint_path}")

    else:
        epochs_without_improvement += 1
        print(f"No improvement for {epochs_without_improvement} epoch(s).")

    if epochs_without_improvement >= PATIENCE:
        print("Early stopping triggered.")
        break


history_df = pd.DataFrame(history)
history_df.to_csv(OUTPUT_DIR / f"training_history_{POOLING}.csv", index=False)

print("Best epoch:", best_epoch)
print("Best validation accuracy:", best_val_acc)


# ============================================================
# 12. Load best model and final validation report
# ============================================================

checkpoint = torch.load(checkpoint_path, map_location=device)

model.load_state_dict(checkpoint["model_state_dict"])
model = model.to(device)

val_loss, val_acc, val_true_ids, val_pred_ids = evaluate(
    model=model,
    dataloader=val_loader,
    criterion=criterion,
    device=device
)

val_true_labels = [id2label[idx] for idx in val_true_ids]
val_pred_labels = [id2label[idx] for idx in val_pred_ids]

print("\nFinal validation accuracy:", val_acc)
print("\nClassification report:")
print(
    classification_report(
        val_true_labels,
        val_pred_labels,
        labels=labels,
        zero_division=0
    )
)

val_predictions_df = pd.DataFrame({
    "Literal": val_split["Literal"].values,
    "Code": val_split["Code"].values,
    "true_y_category": val_true_labels,
    "predicted_y_category": val_pred_labels
})

val_predictions_df.to_csv(
    OUTPUT_DIR / f"validation_predictions_{POOLING}.csv",
    index=False
)


# ============================================================
# 13. Predict leaderboard data
# ============================================================

leaderboard_pred_ids = predict(
    model=model,
    dataloader=leaderboard_loader,
    device=device
)

leaderboard_pred_labels = [id2label[idx] for idx in leaderboard_pred_ids]

submission = pd.DataFrame({
    "id": leaderboard_df["id"],
    "y_category": leaderboard_pred_labels
})

submission_path = OUTPUT_DIR / f"submission_roberta_{POOLING}.csv"
submission.to_csv(submission_path, index=False)

print("\nSaved submission:", submission_path.resolve())
print("File exists:", submission_path.exists())

if submission_path.exists():
    print("File size:", submission_path.stat().st_size, "bytes")

print(submission.head(20))