import os
import random
import math
import torch
import numpy as np
import pandas as pd
from torch.nn import CrossEntropyLoss
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
    DataCollatorWithPadding,
)
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score,
    recall_score, confusion_matrix,
)

# CONFIG

MODEL_NAME = "xlm-roberta-base"
DATASET_PATH = "./dataset/"
OUTPUT_DIR = "./models/cybershield-model"
MAX_LENGTH = 128
MAX_SAMPLES_PER_CLASS = 1500
MAX_PER_FILE = 3000
EPOCHS = 4
BATCH_SIZE = 4
GRAD_ACCUM = 4
LEARNING_RATE = 3e-6
WARMUP_STEPS = 200
WEIGHT_DECAY = 0.01
LABEL_SMOOTHING = 0.05
MAX_GRAD_NORM = 0.3
SEED = 42


random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# DATASET CLASS

class CyberShieldDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {k: torch.tensor(v[idx]) for k, v in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx]).long()
        return item

    def __len__(self):
        return len(self.labels)

# LABEL NORMALIZATION


def normalize_label(x):

    if isinstance(x, float) and math.isnan(x):
        return None

    x = str(x).strip().lower()

    if x in ["0", "0.0", "toxic", "hate", "offensive", "abusive"]:
        return 0
    if x in ["1", "1.0", "clean", "neutral", "non-toxic", "non_toxic",
             "not offensive", "not abusive", "normal", "none"]:
        return 1

    print(f"⚠️  Unknown label '{x}' — row dropped.")
    return None

# LOAD DATA


def load_data(path):
    train_texts, train_labels = [], []
    val_texts,   val_labels = [], []
    test_texts,  test_labels = [], []

    csv_files = [f for f in os.listdir(path) if f.endswith(".csv")]
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in: {path}")

    print(f"\n📂 Found {len(csv_files)} CSV file(s)")

    for file in sorted(csv_files):
        fpath = os.path.join(path, file)
        try:
            df = pd.read_csv(fpath, encoding="utf-8")
        except UnicodeDecodeError:
            try:
                df = pd.read_csv(fpath, encoding="latin-1")
                print(f"  ℹ️  '{file}': latin-1 fallback")
            except Exception as e:
                print(f"⚠️  Skipping '{file}': {e}")
                continue
        except Exception as e:
            print(f"⚠️  Skipping '{file}': {e}")
            continue

        text_col = next(
            (c for c in ["text", "comment", "sentence", "content", "comment_text"]
             if c in df.columns), None)
        label_col = next(
            (c for c in ["label", "toxic", "class"] if c in df.columns), None)

        if not text_col or not label_col:
            print(
                f"⚠️  Skipping '{file}': columns not found {list(df.columns)}")
            continue

        df = df[[text_col, label_col]].copy()
        df.columns = ["text", "label"]
        df = df.dropna(subset=["text"])
        df["text"] = df["text"].astype(str).str.strip()
        df = df[df["text"].str.len() > 0]
        flip = False
        df["label"] = df["label"].apply(normalize_label)
        df = df.dropna(subset=["label"])

        if df.empty:
            print(f"⚠️  Skipping '{file}': no usable rows.")
            continue

        if df["label"].nunique() < 2:
            print(f"⚠️  Skipping '{file}': only one label class.")
            continue

        counts = df["label"].value_counts()
        n_samples = min(MAX_SAMPLES_PER_CLASS, counts.min())
        df = pd.concat([
            df[df["label"] == lbl].sample(n=n_samples, random_state=SEED)
            for lbl in counts.index
        ]).reset_index(drop=True)

        if len(df) > MAX_PER_FILE:
            df = df.sample(
                n=MAX_PER_FILE, random_state=SEED).reset_index(drop=True)

        texts = df["text"].tolist()
        labels = df["label"].astype(int).tolist()

        file_lower = file.lower()

        if "train" in file_lower:
            train_texts += texts
            train_labels += labels
            split = "train"
        elif "val" in file_lower or "dev" in file_lower:
            val_texts += texts
            val_labels += labels
            split = "val"
        elif "test" in file_lower:
            test_texts += texts
            test_labels += labels
            split = "test"
        else:
            print(f"⚠️  Skipping '{file}': no split keyword in filename.")
            continue

        print(f"  ✅ [{split:>5}] {file:<42} "
              f"toxic={labels.count(0):>5} | non-toxic={labels.count(1):>5} | "
              f"total={len(labels):>5}")

    if not train_texts:
        raise ValueError("No training data. Check filenames/columns.")
    if not val_texts:
        raise ValueError("No validation data. Check filenames/columns.")

    print(f"\n✅ DATA LOADED")
    print(f"  Train : {len(train_texts):>6}")
    print(f"  Val   : {len(val_texts):>6}")
    print(f"  Test  : {len(test_texts):>6}")

    return train_texts, train_labels, val_texts, val_labels, test_texts, test_labels

# METRICS


def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "f1": f1_score(labels, preds, average="macro"),
        "precision": precision_score(labels, preds, average="macro"),
        "recall": recall_score(labels, preds, average="macro"),
    }

# TRAIN


def train():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nUsing device: {device}")
    if device == "cuda":
        vram = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"  GPU  : {torch.cuda.get_device_name(0)}")
        print(f"  VRAM : {vram:.1f} GB")
        if vram < 6:
            print("  ⚠️  <6GB VRAM (RTX 3050) — using BATCH_SIZE=4, GRAD_ACCUM=4")
            print("      MAX_PER_FILE capped at 3000 to prevent OOM")

    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        model = AutoModelForSequenceClassification.from_pretrained(
            MODEL_NAME, num_labels=2).to(device)
    except OSError as e:
        print(f"  ⚠️  HuggingFace download failed: {e}")
        print("  🔄 Retrying from local cache...")
        try:
            tokenizer = AutoTokenizer.from_pretrained(
                MODEL_NAME, local_files_only=True)
            model = AutoModelForSequenceClassification.from_pretrained(
                MODEL_NAME, num_labels=2, local_files_only=True).to(device)
            print("  ✅ Loaded from local cache")
        except Exception as exc:
            raise OSError(
                f"Cannot load '{MODEL_NAME}' from HuggingFace or local cache.\n"
                "  Fix options:\n"
                "  1. Wait a few minutes and retry (HF server outage)\n"
                "  2. Set HF_TOKEN: import os; os.environ['HF_TOKEN']='your_token'\n"
                "  3. Pre-download: snapshot_download('xlm-roberta-base')"
            ) from exc

    (train_text, train_labels,
     val_text,   val_labels,
     test_text,  test_labels) = load_data(DATASET_PATH)

    train_enc = tokenizer(train_text, truncation=True, padding=False,
                          max_length=MAX_LENGTH, return_attention_mask=True)
    val_enc = tokenizer(val_text,   truncation=True, padding=False,
                        max_length=MAX_LENGTH, return_attention_mask=True)
    test_enc = tokenizer(test_text,  truncation=True, padding=False,
                         max_length=MAX_LENGTH, return_attention_mask=True)

    train_ds = CyberShieldDataset(train_enc, train_labels)
    val_ds = CyberShieldDataset(val_enc,   val_labels)
    test_ds = CyberShieldDataset(test_enc,  test_labels)
    data_collator = DataCollatorWithPadding(tokenizer)

    label_counts = torch.tensor(
        [train_labels.count(0), train_labels.count(1)], dtype=torch.float)
    imbalance_ratio = abs(
        label_counts[0] - label_counts[1]) / label_counts.sum()

    if imbalance_ratio > 0.1:
        class_weights = (label_counts.sum() / (2 * label_counts)).to(device)
        print(
            f"\n  ⚖️  Imbalance {imbalance_ratio:.1%} — weighted loss applied")
    else:
        class_weights = torch.tensor([1.0, 1.0]).to(device)
        print(f"\n  ✅ Balanced ({imbalance_ratio:.1%}) — equal weights")

    print(
        f"  Weights → TOXIC:{class_weights[0]:.3f} | NON-TOXIC:{class_weights[1]:.3f}")
    class_weights = class_weights.float()

    class CyberShieldTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False,
                         num_items_in_batch=None, **kwargs):
            labels = inputs.get("labels")
            outputs = model(
                **{k: v for k, v in inputs.items() if k != "labels"})
            logits = outputs.get("logits")
            loss = CrossEntropyLoss(
                weight=class_weights,
                label_smoothing=LABEL_SMOOTHING,
            )(logits.float(), labels)

            if num_items_in_batch is not None:
                loss = loss / num_items_in_batch * labels.shape[0]

            return (loss, outputs) if return_outputs else loss

    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        learning_rate=LEARNING_RATE,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM,
        num_train_epochs=EPOCHS,
        warmup_steps=WARMUP_STEPS,
        weight_decay=WEIGHT_DECAY,
        logging_steps=20,
        fp16=(device == "cuda"),
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        max_grad_norm=MAX_GRAD_NORM,
        seed=SEED,
        report_to="none",
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        dataloader_pin_memory=True,
        dataloader_num_workers=0,
    )

    trainer = CyberShieldTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        compute_metrics=compute_metrics,
        data_collator=data_collator,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
    )

    if device == "cuda":
        torch.cuda.empty_cache()

    print("\n🚀 Training CyberShield...\n")
    trainer.train()

    val_out = trainer.predict(val_ds)
    val_probs = torch.softmax(
        torch.tensor(val_out.predictions, dtype=torch.float32), dim=1
    ).numpy()
    np.save("val_probs.npy",  val_probs[:, 0])
    np.save("val_labels.npy", val_out.label_ids)
    print("✅ val_probs.npy + val_labels.npy saved")

    if test_text:
        print("\n📊 FINAL TEST RESULTS")
        results = trainer.predict(test_ds)
        m = results.metrics

        print("=" * 42)
        print(f"  Accuracy  : {m['test_accuracy']:.4f}")
        print(f"  F1        : {m['test_f1']:.4f}")
        print(f"  Precision : {m['test_precision']:.4f}")
        print(f"  Recall    : {m['test_recall']:.4f}")
        print("=" * 42)

        preds = results.predictions.argmax(-1)
        cm = confusion_matrix(test_labels, preds)

        print("\n  Confusion Matrix:")
        print(f"  {'':20} Pred TOXIC  Pred NON-TOXIC")
        print(f"  {'Actual TOXIC':20}   {cm[0][0]:>6}        {cm[0][1]:>6}")
        print(
            f"  {'Actual NON-TOXIC':20}   {cm[1][0]:>6}        {cm[1][1]:>6}")

        row0 = cm[0][0] + cm[0][1]
        row1 = cm[1][0] + cm[1][1]
        fpr = cm[0][1] / row0 if row0 > 0 else float("nan")
        fnr = cm[1][0] / row1 if row1 > 0 else float("nan")

        print(f"\n  FPR: {fpr:.3f}  (safe text flagged toxic)")
        print(f"  FNR: {fnr:.3f}  (toxic text missed)")

        if not np.isnan(fnr) and fnr > 0.2:
            print("  ⚠️  High FNR → lower TOXIC_THRESHOLD in test.py to ~0.4")
        if not np.isnan(fpr) and fpr > 0.2:
            print("  ⚠️  High FPR → raise TOXIC_THRESHOLD in test.py to ~0.6")
        print("=" * 42)
    else:
        print("\n⚠️  No test files — skipping held-out evaluation.")

    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"\n✅ Model saved → {OUTPUT_DIR}")
    print("   Ready for XAI layer.\n")


if __name__ == "__main__":
    train()
