
from pathlib import Path
import json

import numpy as np
import pandas as pd
import torch

from datasets import Dataset
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
    set_seed,
)

ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = ROOT / "data" / "processed" / "transformer"
MODEL_DIR = ROOT / "models" / "text" / "distilbert-phishing"
REPORT_DIR = ROOT / "reports" / "transformer"

CHECKPOINT = "distilbert/distilbert-base-uncased"
MAX_LENGTH = 256
SEED = 42

ID2LABEL = {
    0: "LEGITIMATE",
    1: "PHISHING",
}

LABEL2ID = {
    "LEGITIMATE": 0,
    "PHISHING": 1,
}


def load_split(name):
    path = DATA_DIR / f"{name}.csv"

    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run prepare_transformer_data.py first."
        )

    df = pd.read_csv(path)

    if not {"text", "label"}.issubset(df.columns):
        raise ValueError(f"{name}.csv needs text and label columns.")

    df = df.dropna(subset=["text", "label"]).copy()
    df["text"] = df["text"].astype(str)
    df["label"] = df["label"].astype(int)

    if not set(df["label"].unique()).issubset({0, 1}):
        raise ValueError(f"Invalid labels in {name}.csv")

    return Dataset.from_pandas(
        df[["text", "label"]].reset_index(drop=True),
        preserve_index=False,
    )


def compute_metrics(eval_prediction):
    logits, labels = eval_prediction

    predictions = np.argmax(logits, axis=-1)

    precision, recall, f1, _ = precision_recall_fscore_support(
        labels,
        predictions,
        labels=[0, 1],
        average="binary",
        zero_division=0,
    )

    return {
        "accuracy": float(accuracy_score(labels, predictions)),
        "phishing_precision": float(precision),
        "phishing_recall": float(recall),
        "phishing_f1": float(f1),
    }


def main():
    set_seed(SEED)

    print("Loading prepared datasets...")
    train_data = load_split("train")
    validation_data = load_split("validation")
    test_data = load_split("test")

    print("Training rows:", len(train_data))
    print("Validation rows:", len(validation_data))
    print("Test rows:", len(test_data))

    print("\nLoading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(CHECKPOINT)

    def tokenize_batch(batch):
        return tokenizer(
            batch["text"],
            truncation=True,
            max_length=MAX_LENGTH,
        )

    train_data = train_data.map(
        tokenize_batch,
        batched=True,
        remove_columns=["text"],
    )
    validation_data = validation_data.map(
        tokenize_batch,
        batched=True,
        remove_columns=["text"],
    )
    test_data = test_data.map(
        tokenize_batch,
        batched=True,
        remove_columns=["text"],
    )

    print("\nLoading pretrained classifier...")
    model = AutoModelForSequenceClassification.from_pretrained(
        CHECKPOINT,
        num_labels=2,
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    )

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.parent.mkdir(parents=True, exist_ok=True)

    training_args = TrainingArguments(
        output_dir=str(ROOT / "models" / "text" / "training_checkpoints"),
        learning_rate=2e-5,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=2,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="phishing_f1",
        greater_is_better=True,
        logging_steps=50,
        save_total_limit=2,
        report_to="none",
        seed=SEED,
        fp16=torch.cuda.is_available(),
        dataloader_num_workers=0,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_data,
        eval_dataset=validation_data,
        processing_class=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
        compute_metrics=compute_metrics,
    )

    print("\nStarting fine-tuning...")
    trainer.train()

    print("\nEvaluating on the held-out test set...")
    test_output = trainer.predict(
        test_data,
        metric_key_prefix="test",
    )

    predicted_labels = np.argmax(test_output.predictions, axis=-1)
    true_labels = test_output.label_ids

    report = classification_report(
        true_labels,
        predicted_labels,
        labels=[0, 1],
        target_names=["LEGITIMATE", "PHISHING"],
        output_dict=True,
        zero_division=0,
    )

    matrix = confusion_matrix(
        true_labels,
        predicted_labels,
        labels=[0, 1],
    )

    metrics = {
        key: float(value)
        for key, value in test_output.metrics.items()
    }

    print("\nTest metrics:")
    print(json.dumps(metrics, indent=2))

    print("\nClassification report:")
    print(classification_report(
        true_labels,
        predicted_labels,
        labels=[0, 1],
        target_names=["LEGITIMATE", "PHISHING"],
        zero_division=0,
    ))

    print("\nConfusion matrix (rows=true, columns=predicted):")
    print(matrix)

    # trainer.model is the best validation-selected checkpoint.
    trainer.save_model(str(MODEL_DIR))
    tokenizer.save_pretrained(str(MODEL_DIR))

    with open(REPORT_DIR / "test_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    with open(REPORT_DIR / "classification_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    with open(REPORT_DIR / "confusion_matrix.json", "w", encoding="utf-8") as f:
        json.dump(matrix.tolist(), f, indent=2)

    print(f"\nSaved trained model to: {MODEL_DIR}")
    print(f"Saved evaluation reports to: {REPORT_DIR}")


if __name__ == "__main__":
    main()
