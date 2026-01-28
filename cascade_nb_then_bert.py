"""
cascade_nb_then_bert.py

Cascade pipeline:
  1) Train a fast TF-IDF + Multinomial Naive Bayes classifier on the training set.
  2) For test examples where NB is confident (max predicted prob >= threshold),
     accept NB's prediction.
  3) For remaining (uncertain) test examples, run a BERT classifier (fine-tuned on the same
     training set) and use BERT's prediction for those examples.
  4) Report metrics for:
       - NB alone (on full test set)
       - BERT alone (on full test set)
       - Cascade (NB-when-confident, else BERT)
"""

import os
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from scipy.special import softmax

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset

# -------------------- Config --------------------
# Dataset paths to try (your uploaded dataset is likely here)
POSSIBLE_PATHS = [
    "/mnt/data/topical_chat.csv.xlsx",
    "/content/topical_chat.csv.xlsx",
    "/mnt/data/topical_chat.csv",
    "/mnt/data/data.csv",
    "data.csv",
    "topical_chat.csv.xlsx",
    "topical_chat.csv"
]

# Text/label column name candidates (found in your notebooks)
TEXT_COL_CANDS = ["message", "text", "Tweet", "sentence"]
LABEL_COL_CANDS = ["sentiment", "label", "target"]

# NB confidence threshold (tune this: higher -> fewer BERT calls)
NB_CONF_THRESHOLD = 0.75

# BERT settings
MODEL_NAME = "bert-base-uncased"
MAX_LENGTH = 128
BERT_EPOCHS = 2
BERT_BATCH_SIZE = 8   # lower if running on CPU / low memory
SEED = 42
RANDOM_STATE = 42

# ------------------------------------------------

def find_dataset(paths=POSSIBLE_PATHS):
    for p in paths:
        if os.path.exists(p):
            return p
    return None

def load_dataset(path):
    if path is None:
        raise FileNotFoundError("Dataset not found. Place it in one of the POSSIBLE_PATHS.")
    if path.endswith(".xlsx") or path.endswith(".xls"):
        return pd.read_excel(path)
    else:
        return pd.read_csv(path)

def detect_columns(df):
    text_col = None
    label_col = None
    for c in TEXT_COL_CANDS:
        if c in df.columns:
            text_col = c
            break
    for c in LABEL_COL_CANDS:
        if c in df.columns:
            label_col = c
            break
    if text_col is None or label_col is None:
        raise ValueError(f"Could not auto-detect text/label columns. Columns present: {df.columns.tolist()}")
    return text_col, label_col

def train_naive_bayes(train_texts, train_labels, ngram=(1,2), max_feats=20000):
    tfidf = TfidfVectorizer(max_features=max_feats, ngram_range=ngram)
    X_train = tfidf.fit_transform(train_texts)
    nb = MultinomialNB(alpha=1.0)
    nb.fit(X_train, train_labels)
    return tfidf, nb

def prepare_hf_dataset(texts, labels=None, tokenizer=None):
    d = {"text": texts}
    if labels is not None:
        d["label"] = labels
    ds = Dataset.from_dict(d)
    if tokenizer is None:
        return ds
    def tok(ex):
        return tokenizer(ex["text"], padding="max_length", truncation=True, max_length=MAX_LENGTH)
    ds = ds.map(tok, batched=True, remove_columns=["text"])
    if labels is not None:
        ds.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])
    else:
        ds.set_format(type="torch", columns=["input_ids", "attention_mask"])
    return ds

def main():
    # 1) Load data
    dataset_path = find_dataset()
    print("Using dataset path:", dataset_path)
    df = load_dataset(dataset_path)
    print("Dataset columns:", df.columns.tolist())
    text_col, label_col = detect_columns(df)
    print("Detected text column:", text_col, "label column:", label_col)
    df = df[[text_col, label_col]].dropna().rename(columns={text_col: "text", label_col: "label"}).reset_index(drop=True)

    # 2) Encode labels
    le = LabelEncoder()
    df["label_id"] = le.fit_transform(df["label"])
    num_labels = len(le.classes_)
    print("Detected classes ({}): {}".format(num_labels, list(le.classes_)))

    # 3) Train / Test split
    train_df, test_df = train_test_split(df, test_size=0.2, stratify=df["label_id"], random_state=RANDOM_STATE)
    train_texts = train_df["text"].tolist()
    train_labels = train_df["label_id"].tolist()
    test_texts = test_df["text"].tolist()
    test_labels = test_df["label_id"].tolist()

    # 4) Train Naive Bayes
    print("\nTraining TF-IDF + MultinomialNB ...")
    tfidf, nb = train_naive_bayes(train_texts, train_labels)
    X_test_tfidf = tfidf.transform(test_texts)
    nb_preds = nb.predict(X_test_tfidf)
    nb_proba = nb.predict_proba(X_test_tfidf)  # shape (n_test, num_labels)
    nb_acc = accuracy_score(test_labels, nb_preds)
    print("Naive Bayes accuracy (full test):", nb_acc)
    print("Naive Bayes classification report:")
    print(classification_report(test_labels, nb_preds, target_names=le.classes_))

    # Identify indices where NB is confident vs uncertain
    nb_confidences = nb_proba.max(axis=1)
    confident_mask = nb_confidences >= NB_CONF_THRESHOLD
    uncertain_mask = ~confident_mask
    idx_confident = np.where(confident_mask)[0]
    idx_uncertain = np.where(uncertain_mask)[0]
    print(f"\nNB confident on {len(idx_confident)}/{len(test_texts)} test samples (threshold={NB_CONF_THRESHOLD})")
    print(f"NB uncertain on {len(idx_uncertain)} samples -> these will go to BERT")

    # 5) Fine-tune BERT on the training set
    print("\nPreparing and fine-tuning BERT (may take time) ...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=num_labels)

    # Prepare hf Datasets
    train_ds = prepare_hf_dataset(train_texts, train_labels, tokenizer)
    eval_ds = prepare_hf_dataset(test_texts, test_labels, tokenizer)  # we will still evaluate full test set for BERT baseline

    training_args = TrainingArguments(
        output_dir="./bert_cascade_out",
        num_train_epochs=BERT_EPOCHS,
        per_device_train_batch_size=BERT_BATCH_SIZE,
        per_device_eval_batch_size=BERT_BATCH_SIZE * 2,
        evaluation_strategy="epoch",
        save_strategy="no",
        learning_rate=2e-5,
        logging_steps=100,
        seed=SEED,
        load_best_model_at_end=False,
    )

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=1)
        acc = (preds == labels).mean()
        return {"accuracy": acc}

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        compute_metrics=compute_metrics,
    )

    trainer.train()

    # BERT baseline on full test set
    print("\nGetting BERT predictions on full test set ...")
    bert_out_full = trainer.predict(eval_ds)
    bert_logits_full = bert_out_full.predictions
    bert_proba_full = softmax(bert_logits_full, axis=1)
    bert_preds_full = np.argmax(bert_logits_full, axis=1)
    bert_acc = accuracy_score(test_labels, bert_preds_full)
    print("BERT accuracy (full test):", bert_acc)
    print("BERT classification report (full test):")
    print(classification_report(test_labels, bert_preds_full, target_names=le.classes_))

    # 6) For cascade: run BERT only on uncertain test samples
    if len(idx_uncertain) > 0:
        print(f"\nRunning BERT on {len(idx_uncertain)} uncertain test samples...")
        uncertain_texts = [test_texts[i] for i in idx_uncertain]
        # Prepare hf dataset without labels (but we have labels to evaluate)
        uncertain_ds = prepare_hf_dataset(uncertain_texts, None, tokenizer)
        # Use Trainer.predict to get logits for these
        bert_uncertain_out = trainer.predict(uncertain_ds)
        bert_uncertain_logits = bert_uncertain_out.predictions
        bert_uncertain_proba = softmax(bert_uncertain_logits, axis=1)
        bert_uncertain_preds = np.argmax(bert_uncertain_logits, axis=1)

        # Build final cascade predictions: start with NB preds, replace uncertain indices with BERT preds
        cascade_preds = nb_preds.copy()
        for local_i, global_i in enumerate(idx_uncertain):
            cascade_preds[global_i] = int(bert_uncertain_preds[local_i])

        cascade_acc = accuracy_score(test_labels, cascade_preds)
        print("\nCascade accuracy (NB if confident else BERT on uncertain):", cascade_acc)
        print("Cascade classification report:")
        print(classification_report(test_labels, cascade_preds, target_names=le.classes_))

        # Also print breakdown of errors for curiousity
        # How many of NB-confident were correct vs incorrect?
        if len(idx_confident) > 0:
            nb_conf_correct = (nb_preds[idx_confident] == np.array(test_labels)[idx_confident]).sum()
            nb_conf_total = len(idx_confident)
            print(f"\nNB-confident: {nb_conf_correct}/{nb_conf_total} correct (accuracy on confident subset: {nb_conf_correct/nb_conf_total:.4f})")

    else:
        print("NB was confident on all samples (no BERT calls needed). Cascade == NB.")

    # 7) Optional: Show sample-level decisions (for debugging)
    # Print a few examples where NB was uncertain and BERT corrected or disagreed
    n_show = 10
    print("\nSample comparisons (where NB uncertain):")
    shown = 0
    for local_i, global_i in enumerate(idx_uncertain):
        nb_p = nb_preds[global_i]
        nb_conf = nb_confidences[global_i]
        bert_p = int(bert_uncertain_preds[local_i])
        true = test_labels[global_i]
        if nb_p != bert_p or nb_p != true:  # show cases where models differ or wrong
            print("-" * 40)
            print("Text:", test_texts[global_i][:300])
            print("True label:", le.inverse_transform([true])[0])
            print(f"NB pred: {le.inverse_transform([nb_p])[0]} (conf={nb_conf:.3f})")
            print("BERT pred:", le.inverse_transform([bert_p])[0])
            shown += 1
            if shown >= n_show:
                break

if __name__ == "__main__":
    main()
