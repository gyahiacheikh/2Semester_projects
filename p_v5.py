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

from transformers import AutoTokenizer, AutoModel, get_linear_schedule_with_warmup
from transformers import logging as transformers_logging

warnings.filterwarnings("ignore")
transformers_logging.set_verbosity_error()

try:
    BASE_DIR = Path(__file__).resolve().parent
except NameError:
    BASE_DIR = Path.cwd().resolve()

TRAIN_PATH = BASE_DIR / "codification_data.csv"
LEADERBOARD_PATH = BASE_DIR / "leaderboard_data.csv"

OUTPUT_DIR = BASE_DIR / "outputs_roberta_mean_v5"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_NAME = "PlanTL-GOB-ES/roberta-base-biomedical-clinical-es"

VALIDATION_SEED = 42

MAX_EPOCHS = 50
PATIENCE = 10
BATCH_SIZE = 128          
LEARNING_RATE = 2e-5
WEIGHT_DECAY = 0.01
DROPOUT = 0.1
MAX_LENGTH = 64

POOLING = "mean"
NUM_WORKERS = 0
WARMUP_RATIO = 0.06
MAX_GRAD_NORM = 1.0

FINAL_SEEDS = [42, 123, 2026]


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


def preprocess_text(text):
    if pd.isna(text):
        return ""

    text = str(text)
    text = text.strip()
    text = re.sub(r"\s+", " ", text)

    return text


def check_file_exists(path, name):
    if not path.exists():
        raise FileNotFoundError(
            f"{name} not found at: {path.resolve()}\n"
            f"Make sure the file is in the same folder as this Python script."
        )


def save_csv_checked(df, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8")

    if not path.exists():
        raise RuntimeError(f"File was not saved correctly: {path.resolve()}")

    print(f"Saved: {path.resolve()}")
    print(f"File size: {path.stat().st_size} bytes")


check_file_exists(TRAIN_PATH, "codification_data.csv")
check_file_exists(LEADERBOARD_PATH, "leaderboard_data.csv")

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

required_train_cols = {"Code", "Literal"}
required_leaderboard_cols = {"id", "Literal"}

missing_train = required_train_cols - set(train_df.columns)
missing_leaderboard = required_leaderboard_cols - set(leaderboard_df.columns)

if missing_train:
    raise ValueError(f"Missing required training columns: {missing_train}")

if missing_leaderboard:
    raise ValueError(f"Missing required leaderboard columns: {missing_leaderboard}")

leaderboard_df["Literal_original"] = leaderboard_df["Literal"].fillna("").astype(str)

train_df["Code"] = train_df["Code"].fillna("").astype(str).str.strip()
train_df = train_df[train_df["Code"].str.len() > 0].copy()

train_df["Literal"] = train_df["Literal"].apply(preprocess_text)
leaderboard_df["Literal"] = leaderboard_df["Literal"].apply(preprocess_text)

train_df["y_category"] = train_df["Code"].astype(str).str[0]

labels = sorted(train_df["y_category"].unique())
label2id = {label: idx for idx, label in enumerate(labels)}
id2label = {idx: label for label, idx in label2id.items()}

train_df["label_id"] = train_df["y_category"].map(label2id)

print("Training data:", train_df.shape)
print("Leaderboard data:", leaderboard_df.shape)
print("Number of classes:", len(labels))
print("Classes:", labels)


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


class ICDRobertaClassifier(nn.Module):
    def __init__(self, model_name, num_classes, pooling="mean", dropout=0.1):
        super().__init__()

        if pooling not in {"cls", "mean"}:
            raise ValueError("pooling must be 'cls' or 'mean'")

        self.pooling = pooling

        self.backbone = AutoModel.from_pretrained(
            model_name,
            add_pooling_layer=False
        )

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


def make_train_loader(df, tokenizer, batch_size, device, shuffle):
    dataset = ICDDataset(
        texts=df["Literal"],
        labels=df["label_id"],
        tokenizer=tokenizer,
        max_length=MAX_LENGTH
    )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=NUM_WORKERS,
        pin_memory=device.type == "cuda"
    )


def make_prediction_loader(df, tokenizer, batch_size, device):
    dataset = ICDDataset(
        texts=df["Literal"],
        labels=None,
        tokenizer=tokenizer,
        max_length=MAX_LENGTH
    )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=device.type == "cuda"
    )


def build_optimizer_and_scheduler(model, train_loader, epochs):
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY
    )

    total_steps = len(train_loader) * epochs
    warmup_steps = int(total_steps * WARMUP_RATIO)

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps
    )

    return optimizer, scheduler


def train_one_epoch(model, dataloader, optimizer, scheduler, criterion, device):
    model.train()

    total_loss = 0.0
    total_examples = 0
    total_correct = 0

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
        torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
        optimizer.step()
        scheduler.step()

        preds = torch.argmax(logits, dim=1)

        batch_size = labels_batch.size(0)
        total_loss += loss.item() * batch_size
        total_examples += batch_size
        total_correct += (preds == labels_batch).sum().item()

    train_loss = total_loss / total_examples
    train_acc = total_correct / total_examples

    return train_loss, train_acc


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


def predict_logits(model, dataloader, device):
    model.eval()

    all_logits = []

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            logits = model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )

            all_logits.append(logits.cpu())

    return torch.cat(all_logits, dim=0).numpy()


def train_with_validation(tokenizer, device):
    set_seed(VALIDATION_SEED)

    class_counts = train_df["label_id"].value_counts()

    if class_counts.min() >= 2:
        stratify_values = train_df["label_id"]
    else:
        stratify_values = None
        print("Warning: stratified split disabled because at least one class has fewer than 2 examples.")

    train_split, val_split = train_test_split(
        train_df,
        test_size=0.2,
        random_state=VALIDATION_SEED,
        stratify=stratify_values
    )

    train_loader = make_train_loader(
        train_split,
        tokenizer,
        BATCH_SIZE,
        device,
        shuffle=True
    )

    val_loader = make_train_loader(
        val_split,
        tokenizer,
        BATCH_SIZE,
        device,
        shuffle=False
    )

    model = ICDRobertaClassifier(
        model_name=MODEL_NAME,
        num_classes=len(labels),
        pooling=POOLING,
        dropout=DROPOUT
    ).to(device)

    criterion = nn.CrossEntropyLoss()

    optimizer, scheduler = build_optimizer_and_scheduler(
        model,
        train_loader,
        MAX_EPOCHS
    )

    best_val_acc = -1.0
    best_epoch = 0
    epochs_without_improvement = 0
    best_state_dict = None

    history = []

    checkpoint_path = OUTPUT_DIR / "validation_best_roberta_mean_v5.pt"

    for epoch in range(1, MAX_EPOCHS + 1):
        train_loss, train_acc = train_one_epoch(
            model=model,
            dataloader=train_loader,
            optimizer=optimizer,
            scheduler=scheduler,
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
            "train_accuracy": train_acc,
            "val_loss": val_loss,
            "val_accuracy": val_acc
        })

        print(
            f"Validation | epoch={epoch:02d} | "
            f"train_loss={train_loss:.4f} | "
            f"train_acc={train_acc:.4f} | "
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

            print(f"Saved best validation model: {checkpoint_path.resolve()}")

        else:
            epochs_without_improvement += 1
            print(f"No improvement for {epochs_without_improvement} epoch(s).")

        if epochs_without_improvement >= PATIENCE:
            print("Early stopping triggered.")
            break

    history_df = pd.DataFrame(history)

    save_csv_checked(
        history_df,
        OUTPUT_DIR / "training_history_roberta_mean_v5.csv"
    )

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)

    val_loss, val_acc, val_true_ids, val_pred_ids = evaluate(
        model=model,
        dataloader=val_loader,
        criterion=criterion,
        device=device
    )

    val_true_labels = [id2label[int(idx)] for idx in val_true_ids]
    val_pred_labels = [id2label[int(idx)] for idx in val_pred_ids]

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

    save_csv_checked(
        val_predictions_df,
        OUTPUT_DIR / "validation_predictions_roberta_mean_v5.csv"
    )

    return best_epoch, best_val_acc


def train_full_model(seed, epochs, tokenizer, device):
    set_seed(seed)

    full_loader = make_train_loader(
        train_df,
        tokenizer,
        BATCH_SIZE,
        device,
        shuffle=True
    )

    model = ICDRobertaClassifier(
        model_name=MODEL_NAME,
        num_classes=len(labels),
        pooling=POOLING,
        dropout=DROPOUT
    ).to(device)

    criterion = nn.CrossEntropyLoss()

    optimizer, scheduler = build_optimizer_and_scheduler(
        model,
        full_loader,
        epochs
    )

    for epoch in range(1, epochs + 1):
        train_loss, train_acc = train_one_epoch(
            model=model,
            dataloader=full_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            criterion=criterion,
            device=device
        )

        print(
            f"Full train | seed={seed} | "
            f"epoch={epoch:02d}/{epochs:02d} | "
            f"train_loss={train_loss:.4f} | "
            f"train_acc={train_acc:.4f}"
        )

    checkpoint_path = OUTPUT_DIR / f"full_model_roberta_mean_v5_seed{seed}.pt"

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "model_name": MODEL_NAME,
            "pooling": POOLING,
            "seed": seed,
            "labels": labels,
            "label2id": label2id,
            "id2label": id2label,
            "max_length": MAX_LENGTH,
            "epochs": epochs
        },
        checkpoint_path
    )

    print(f"Saved full model: {checkpoint_path.resolve()}")

    return model


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

set_seed(VALIDATION_SEED)

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

best_epoch, best_val_acc = train_with_validation(
    tokenizer=tokenizer,
    device=device
)

print("Best validation epoch:", best_epoch)
print("Best validation accuracy:", best_val_acc)

leaderboard_loader = make_prediction_loader(
    leaderboard_df,
    tokenizer,
    BATCH_SIZE,
    device
)

all_leaderboard_logits = []

for seed in FINAL_SEEDS:
    model = train_full_model(
        seed=seed,
        epochs=best_epoch,
        tokenizer=tokenizer,
        device=device
    )

    logits = predict_logits(
        model=model,
        dataloader=leaderboard_loader,
        device=device
    )

    all_leaderboard_logits.append(logits)

    del model

    if device.type == "cuda":
        torch.cuda.empty_cache()

mean_logits = np.mean(
    np.stack(all_leaderboard_logits, axis=0),
    axis=0
)

leaderboard_pred_ids = np.argmax(mean_logits, axis=1)

leaderboard_pred_labels = [
    str(id2label[int(idx)]) for idx in leaderboard_pred_ids
]

submission = pd.DataFrame({
    "id": leaderboard_df["id"],
    "Literal": leaderboard_df["Literal_original"],
    "y_category": leaderboard_pred_labels
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
    raise ValueError("Submission contains empty y_category values.")

if len(submission) != len(leaderboard_df):
    raise ValueError(
        f"Submission row count mismatch: submission={len(submission)}, "
        f"leaderboard={len(leaderboard_df)}"
    )

submission_path = OUTPUT_DIR / "submission_roberta_mean_v5.csv"

save_csv_checked(
    submission,
    submission_path
)

print("\nSubmission preview:")
print(submission.head(20))