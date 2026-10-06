# raw_vgrf_1dcnn_baseline.py
# ============================================================
# Non-GAF baseline for Parkinson's disease classification
# Input: 16 sensor-specific VGRF signals x 1024 time points
# Model: conventional 1D-CNN
#
# Protocol:
# - Each VGRF channel independently resampled to 1024 points
# - Per-recording/per-channel z-score normalization
# - Subject-level 80/20 development/test split
# - Reuses gait_gaf_ablation subject split when available
# - Subject-wise stratified 5-fold CV within development cohort
# - Fold selection by subject-level validation AUC
# - Final epochs = median best epoch across 5 folds
# - Subject prediction = mean probability across recordings
# - Threshold = 0.50
# - 2000 stratified subject-level percentile bootstrap 95% CI
#
# This script intentionally DOES NOT use GAF.
# ============================================================

import os
import re
import glob
import gc
import json
import time
import random
import platform

import numpy as np
import pandas as pd
from tqdm import tqdm

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    roc_auc_score,
    confusion_matrix,
)

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


# ============================================================
# 1. Settings
# ============================================================
DATA_ROOT = r"./gait-in-parkinsons-disease-1.0.0"
OUTPUT_ROOT = r"./raw_vgrf_1dcnn_baseline"

# Existing GAF ablation output. Reusing this split makes the
# raw-vs-GAF comparison paired on exactly the same test subjects.
GAF_ABLATION_ROOT = r"./gait_gaf_ablation"
REUSE_GAF_SPLIT = True

SIGNAL_LENGTH = 1024
N_CHANNELS = 16

TEST_SIZE = 0.20
N_FOLDS = 5
SEED = 40

EPOCHS = 80
PATIENCE = 15
LR = 5e-6
WEIGHT_DECAY = 1e-2
LABEL_SMOOTHING = 0.05

BATCH_SIZE = 16
GRAD_ACCUM_STEPS = 1
NUM_WORKERS = 0

SUBJECT_THRESHOLD = 0.50
BOOTSTRAP_ITERATIONS = 2000
BOOTSTRAP_ALPHA = 0.05


# ============================================================
# 2. Device / reproducibility
# ============================================================
def select_device():
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


DEVICE = select_device()


def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


# ============================================================
# 3. Labels / subject IDs
# ============================================================
def get_label_from_filename(filename):
    name = os.path.basename(filename)
    if "Co" in name:
        return 0
    if "Pt" in name:
        return 1
    return None


def get_subject_id(filename):
    name = os.path.splitext(os.path.basename(filename))[0]
    match = re.search(r"(Ga|Ju|Si)(Co|Pt)\d+", name)
    if match:
        return match.group(0)
    return name.split("_")[0]


# ============================================================
# 4. Read the 16 sensor-specific VGRF channels
# Expected columns:
# time + L1-L8 + R1-R8 + total left/right force
# ============================================================
def read_16_sensors(file_path):
    df = pd.read_csv(
        file_path,
        sep=r"\s+",
        header=None,
        engine="python",
    )

    df = df.apply(pd.to_numeric, errors="coerce")
    df = df.dropna(axis=1, how="all")
    df = df.dropna(axis=0, how="any")

    data = df.values

    if data.shape[1] < 19:
        raise ValueError(
            f"Expected at least 19 columns, got {data.shape[1]}: {file_path}"
        )

    # Columns 1:17 = L1-L8 + R1-R8.
    return data[:, 1:17].astype(np.float32)


# ============================================================
# 5. Raw VGRF preprocessing
# This matches the preprocessing immediately BEFORE GAF creation:
# raw signal -> linear interpolation to 1024 -> z-score
# ============================================================
def preprocess_16_sensors(sensors, length=SIGNAL_LENGTH):
    if sensors is None or sensors.shape[0] < 10 or sensors.shape[1] != 16:
        return None

    old_x = np.linspace(0, 1, sensors.shape[0])
    new_x = np.linspace(0, 1, length)

    processed = []

    for i in range(16):
        sig = sensors[:, i].astype(np.float32)

        # Standardize temporal length.
        sig = np.interp(new_x, old_x, sig)

        # Independent channel-wise z-score standardization.
        sig = (sig - np.mean(sig)) / (np.std(sig) + 1e-8)

        processed.append(sig)

    # [channels, time] = [16, 1024]
    return np.stack(processed, axis=0).astype(np.float32)


# ============================================================
# 6. Build raw-VGRF arrays
# ============================================================
def create_raw_dataset():
    os.makedirs(OUTPUT_ROOT, exist_ok=True)
    array_dir = os.path.join(OUTPUT_ROOT, "arrays")
    os.makedirs(array_dir, exist_ok=True)

    files = sorted(
        glob.glob(
            os.path.join(DATA_ROOT, "**", "*.txt"),
            recursive=True,
        )
    )

    if not files:
        raise FileNotFoundError(
            f"No .txt gait files found under DATA_ROOT={DATA_ROOT}"
        )

    rows = []
    exclusions = []

    for file_path in tqdm(files, desc="Creating raw VGRF arrays"):
        label = get_label_from_filename(file_path)
        if label is None:
            continue

        try:
            sensors = read_16_sensors(file_path)
            x = preprocess_16_sensors(sensors)

            if x is None:
                exclusions.append({
                    "source_file": file_path,
                    "reason": "invalid or too-short 16-sensor signal",
                })
                continue

            base = os.path.splitext(os.path.basename(file_path))[0]
            subject_id = get_subject_id(file_path)
            save_path = os.path.join(array_dir, base + ".npy")

            np.save(save_path, x)

            rows.append({
                "array_path": save_path,
                "label": label,
                "label_name": "PD" if label == 1 else "Control",
                "subject_id": subject_id,
                "source_file": file_path,
                "representation": "raw_16ch_vgrf",
                "in_chans": N_CHANNELS,
                "signal_length": SIGNAL_LENGTH,
            })

        except Exception as e:
            exclusions.append({
                "source_file": file_path,
                "reason": str(e),
            })

    meta = pd.DataFrame(rows)

    if meta.empty:
        raise RuntimeError("No eligible raw VGRF recordings were created.")

    meta.to_csv(
        os.path.join(OUTPUT_ROOT, "metadata.csv"),
        index=False,
    )

    pd.DataFrame(
        exclusions,
        columns=["source_file", "reason"],
    ).to_csv(
        os.path.join(OUTPUT_ROOT, "exclusions.csv"),
        index=False,
    )

    return meta


# ============================================================
# 7. Subject-level development / independent test split
# ============================================================
def load_or_create_subject_split(meta):
    gaf_dev_path = os.path.join(
        GAF_ABLATION_ROOT,
        "development_subjects.csv",
    )
    gaf_test_path = os.path.join(
        GAF_ABLATION_ROOT,
        "independent_test_subjects.csv",
    )

    if (
        REUSE_GAF_SPLIT
        and os.path.exists(gaf_dev_path)
        and os.path.exists(gaf_test_path)
    ):
        print("\nReusing subject split from the GAF ablation experiment.")

        dev_subjects = pd.read_csv(gaf_dev_path)
        test_subjects = pd.read_csv(gaf_test_path)

        raw_subject_ids = set(meta["subject_id"].unique())
        expected = (
            set(dev_subjects["subject_id"])
            | set(test_subjects["subject_id"])
        )

        missing = sorted(expected - raw_subject_ids)
        if missing:
            raise RuntimeError(
                "Some subjects from the GAF split are missing from the raw "
                f"baseline dataset: {missing}"
            )

    else:
        print("\nGAF split not found. Recreating the same 80/20 split with seed=40.")

        subject_df = (
            meta[["subject_id", "label"]]
            .drop_duplicates()
            .sort_values("subject_id")
            .reset_index(drop=True)
        )

        dev_subjects, test_subjects = train_test_split(
            subject_df,
            test_size=TEST_SIZE,
            random_state=SEED,
            stratify=subject_df["label"],
        )

        dev_subjects = dev_subjects.reset_index(drop=True)
        test_subjects = test_subjects.reset_index(drop=True)

    dev_subjects.to_csv(
        os.path.join(OUTPUT_ROOT, "development_subjects.csv"),
        index=False,
    )
    test_subjects.to_csv(
        os.path.join(OUTPUT_ROOT, "independent_test_subjects.csv"),
        index=False,
    )

    return dev_subjects, test_subjects


def make_folds(dev_subjects):
    skf = StratifiedKFold(
        n_splits=N_FOLDS,
        shuffle=True,
        random_state=SEED,
    )

    X = dev_subjects["subject_id"].values
    y = dev_subjects["label"].values

    folds = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), start=1):
        folds.append({
            "fold": fold,
            "train_subject_ids": set(
                dev_subjects.iloc[train_idx]["subject_id"]
            ),
            "val_subject_ids": set(
                dev_subjects.iloc[val_idx]["subject_id"]
            ),
        })

    return folds


# ============================================================
# 8. Dataset / DataLoader
# ============================================================
class RawVGRFDataset(Dataset):
    def __init__(self, df):
        self.df = df.reset_index(drop=True)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        x = np.load(self.df.loc[idx, "array_path"]).astype(np.float32)
        y = int(self.df.loc[idx, "label"])

        return (
            torch.from_numpy(x),
            torch.tensor(y, dtype=torch.long),
        )


def make_loader(df, shuffle):
    return DataLoader(
        RawVGRFDataset(df),
        batch_size=BATCH_SIZE,
        shuffle=shuffle,
        num_workers=NUM_WORKERS,
        pin_memory=(DEVICE == "cuda"),
    )


# ============================================================
# 9. Conventional raw-VGRF 1D-CNN
# Input shape: [batch, 16, 1024]
# ============================================================
class RawVGRF1DCNN(nn.Module):
    def __init__(self, in_channels=N_CHANNELS, num_classes=2):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv1d(
                in_channels, 64,
                kernel_size=7, stride=1, padding=3, bias=False
            ),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2),

            nn.Conv1d(
                64, 128,
                kernel_size=5, stride=1, padding=2, bias=False
            ),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2),

            nn.Conv1d(
                128, 256,
                kernel_size=3, stride=1, padding=1, bias=False
            ),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),

            nn.AdaptiveAvgPool1d(1),
        )

        self.classifier = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = x.squeeze(-1)
        return self.classifier(x)


def build_model():
    return RawVGRF1DCNN().to(DEVICE)


def count_parameters(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(
        p.numel() for p in model.parameters()
        if p.requires_grad
    )
    return total, trainable


# ============================================================
# 10. Metrics
# ============================================================
def calculate_metrics(y_true, y_pred, y_prob, loss=np.nan):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    y_prob = np.asarray(y_prob)

    accuracy = accuracy_score(y_true, y_pred)

    precision, sensitivity, f1, _ = (
        precision_recall_fscore_support(
            y_true,
            y_pred,
            average="binary",
            zero_division=0,
        )
    )

    try:
        auc = roc_auc_score(y_true, y_prob)
    except Exception:
        auc = np.nan

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    specificity = tn / (tn + fp + 1e-8)

    return {
        "loss": loss,
        "accuracy": accuracy,
        "precision": precision,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "f1": f1,
        "auc": auc,
        "confusion_matrix": cm,
        "y_true": y_true.tolist(),
        "y_pred": y_pred.tolist(),
        "y_prob": y_prob.tolist(),
    }


def evaluate_recording_level(model, loader, criterion):
    model.eval()

    total_loss = 0.0
    y_true, y_pred, y_prob = [], [], []

    with torch.no_grad():
        for x, labels in loader:
            x = x.to(DEVICE, non_blocking=True)
            labels = labels.to(DEVICE, non_blocking=True)

            outputs = model(x)
            loss = criterion(outputs, labels)

            probs = torch.softmax(outputs, dim=1)[:, 1]
            preds = torch.argmax(outputs, dim=1)

            total_loss += loss.item()

            y_true.extend(labels.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())
            y_prob.extend(probs.cpu().numpy())

    return calculate_metrics(
        y_true,
        y_pred,
        y_prob,
        total_loss / max(len(loader), 1),
    )


def aggregate_to_subject_level(recording_df):
    subject_df = (
        recording_df.groupby(
            ["subject_id", "label", "label_name"],
            as_index=False,
        )
        .agg(
            prob_PD=("prob_PD", "mean"),
            n_recordings=("prob_PD", "size"),
        )
    )

    subject_df["y_true"] = subject_df["label"].astype(int)
    subject_df["y_pred"] = (
        subject_df["prob_PD"] >= SUBJECT_THRESHOLD
    ).astype(int)

    return subject_df


def evaluate_subject_level(subject_df):
    return calculate_metrics(
        subject_df["y_true"].values,
        subject_df["y_pred"].values,
        subject_df["prob_PD"].values,
    )


# ============================================================
# 11. Stratified subject-level percentile bootstrap
# ============================================================
def bootstrap_subject_metrics(subject_df):
    rng = np.random.default_rng(SEED)

    control = subject_df[
        subject_df["y_true"] == 0
    ].reset_index(drop=True)

    pd_group = subject_df[
        subject_df["y_true"] == 1
    ].reset_index(drop=True)

    metric_names = [
        "accuracy",
        "precision",
        "sensitivity",
        "specificity",
        "f1",
        "auc",
    ]

    values = {m: [] for m in metric_names}

    for _ in range(BOOTSTRAP_ITERATIONS):
        b0 = control.iloc[
            rng.integers(0, len(control), len(control))
        ]
        b1 = pd_group.iloc[
            rng.integers(0, len(pd_group), len(pd_group))
        ]

        boot = pd.concat([b0, b1], ignore_index=True)

        m = calculate_metrics(
            boot["y_true"],
            boot["y_pred"],
            boot["prob_PD"],
        )

        for name in metric_names:
            if not np.isnan(m[name]):
                values[name].append(m[name])

    lower_q = 100 * BOOTSTRAP_ALPHA / 2
    upper_q = 100 * (1 - BOOTSTRAP_ALPHA / 2)

    result = {}

    for name, vals in values.items():
        result[f"{name}_ci_low"] = float(
            np.percentile(vals, lower_q)
        )
        result[f"{name}_ci_high"] = float(
            np.percentile(vals, upper_q)
        )

    return result


# ============================================================
# 12. Weighted cross-entropy
# ============================================================
def make_criterion(train_df):
    counts = train_df["label"].value_counts().sort_index()

    weights = np.asarray(
        [
            1.0 / max(counts.get(0, 1), 1),
            1.0 / max(counts.get(1, 1), 1),
        ],
        dtype=np.float32,
    )

    weights = weights / weights.sum() * 2.0

    return nn.CrossEntropyLoss(
        weight=torch.tensor(
            weights,
            dtype=torch.float32,
            device=DEVICE,
        ),
        label_smoothing=LABEL_SMOOTHING,
    )


# ============================================================
# 13. Train one epoch
# ============================================================
def train_one_epoch(model, loader, criterion, optimizer):
    model.train()
    total_loss = 0.0

    optimizer.zero_grad(set_to_none=True)

    for step, (x, labels) in enumerate(loader, start=1):
        x = x.to(DEVICE, non_blocking=True)
        labels = labels.to(DEVICE, non_blocking=True)

        outputs = model(x)
        loss = criterion(outputs, labels) / GRAD_ACCUM_STEPS
        loss.backward()

        if (
            step % GRAD_ACCUM_STEPS == 0
            or step == len(loader)
        ):
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        total_loss += loss.item() * GRAD_ACCUM_STEPS

    return total_loss / max(len(loader), 1)


# ============================================================
# 14. Development 5-fold CV
# ============================================================
def train_cv_fold(fold_number, train_df, val_df, fold_dir):
    set_seed(SEED + fold_number)

    train_loader = make_loader(train_df, True)
    val_loader = make_loader(val_df, False)

    criterion = make_criterion(train_df)
    model = build_model()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LR,
        weight_decay=WEIGHT_DECAY,
    )

    best_auc = -1.0
    best_epoch = 0
    no_improve = 0
    epoch_logs = []

    for epoch in range(1, EPOCHS + 1):
        train_loss = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
        )

        val_recording_result = evaluate_recording_level(
            model,
            val_loader,
            criterion,
        )

        val_pred_df = val_df.copy().reset_index(drop=True)
        val_pred_df["prob_PD"] = val_recording_result["y_prob"]

        val_subject_df = aggregate_to_subject_level(val_pred_df)
        val_subject_result = evaluate_subject_level(val_subject_df)

        current_auc = (
            val_subject_result["auc"]
            if not np.isnan(val_subject_result["auc"])
            else -1.0
        )

        epoch_logs.append({
            "representation": "raw_16ch_vgrf",
            "model": "1D-CNN",
            "fold": fold_number,
            "epoch": epoch,
            "train_loss": train_loss,
            "validation_subject_auc": current_auc,
            "validation_subject_accuracy":
                val_subject_result["accuracy"],
            "validation_subject_f1":
                val_subject_result["f1"],
        })

        print(
            f"Raw VGRF | 1D-CNN | Fold {fold_number} | "
            f"Epoch {epoch:03d} | "
            f"Subject Val AUC={current_auc:.4f}"
        )

        if current_auc > best_auc:
            best_auc = current_auc
            best_epoch = epoch
            no_improve = 0

            torch.save(
                model.state_dict(),
                os.path.join(fold_dir, "best_model.pth"),
            )
        else:
            no_improve += 1

        if no_improve >= PATIENCE:
            break

    result = {
        "representation": "raw_16ch_vgrf",
        "model": "1D-CNN",
        "fold": fold_number,
        "train_subjects": train_df["subject_id"].nunique(),
        "validation_subjects": val_df["subject_id"].nunique(),
        "train_recordings": len(train_df),
        "validation_recordings": len(val_df),
        "best_epoch": best_epoch,
        "best_validation_subject_auc": best_auc,
    }

    del model
    gc.collect()

    if DEVICE == "cuda":
        torch.cuda.empty_cache()

    return result, epoch_logs


# ============================================================
# 15. Final development training + untouched test
# ============================================================
def train_final_and_test(development_df, test_df, final_epochs):
    print("\n" + "=" * 80)
    print(
        f"FINAL | Raw 16-channel VGRF | 1D-CNN | "
        f"16x{SIGNAL_LENGTH} | epochs={final_epochs}"
    )
    print("=" * 80)

    set_seed(SEED)

    train_loader = make_loader(development_df, True)
    test_loader = make_loader(test_df, False)

    criterion = make_criterion(development_df)
    model = build_model()
    total_params, trainable_params = count_parameters(model)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LR,
        weight_decay=WEIGHT_DECAY,
    )

    if DEVICE == "cuda":
        torch.cuda.reset_peak_memory_stats()

    start = time.time()

    for epoch in range(1, final_epochs + 1):
        loss = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
        )

        print(
            f"Final epoch {epoch:03d}/{final_epochs} | "
            f"Loss={loss:.4f}"
        )

    training_time_sec = time.time() - start

    peak_gpu_memory_mb = np.nan
    if DEVICE == "cuda":
        peak_gpu_memory_mb = (
            torch.cuda.max_memory_allocated() / (1024 ** 2)
        )
        torch.cuda.synchronize()

    inference_start = time.time()

    recording_result = evaluate_recording_level(
        model,
        test_loader,
        criterion,
    )

    if DEVICE == "cuda":
        torch.cuda.synchronize()

    inference_time_sec = time.time() - inference_start
    inference_ms_per_recording = (
        1000.0 * inference_time_sec / max(len(test_df), 1)
    )

    pred_df = test_df.copy().reset_index(drop=True)
    pred_df["y_true"] = recording_result["y_true"]
    pred_df["y_pred"] = recording_result["y_pred"]
    pred_df["prob_PD"] = recording_result["y_prob"]

    subject_df = aggregate_to_subject_level(pred_df)
    subject_result = evaluate_subject_level(subject_df)
    ci = bootstrap_subject_metrics(subject_df)

    pred_df.to_csv(
        os.path.join(
            OUTPUT_ROOT,
            "independent_test_recording_predictions.csv",
        ),
        index=False,
    )

    subject_df.to_csv(
        os.path.join(
            OUTPUT_ROOT,
            "independent_test_subject_predictions.csv",
        ),
        index=False,
    )

    torch.save(
        model.state_dict(),
        os.path.join(OUTPUT_ROOT, "final_model.pth"),
    )

    result = {
        "representation": "raw_16ch_vgrf",
        "model": "1D-CNN",
        "input_channels": N_CHANNELS,
        "signal_length": SIGNAL_LENGTH,
        "final_training_epochs": final_epochs,

        "development_subjects":
            development_df["subject_id"].nunique(),
        "development_recordings": len(development_df),
        "independent_test_subjects":
            test_df["subject_id"].nunique(),
        "independent_test_recordings": len(test_df),

        "subject_test_accuracy": subject_result["accuracy"],
        "subject_test_precision": subject_result["precision"],
        "subject_test_sensitivity": subject_result["sensitivity"],
        "subject_test_specificity": subject_result["specificity"],
        "subject_test_f1": subject_result["f1"],
        "subject_test_auc": subject_result["auc"],

        **ci,

        "subject_confusion_matrix": json.dumps(
            subject_result["confusion_matrix"].tolist()
        ),

        "recording_test_accuracy":
            recording_result["accuracy"],
        "recording_test_f1":
            recording_result["f1"],
        "recording_test_auc":
            recording_result["auc"],

        "total_parameters": total_params,
        "trainable_parameters": trainable_params,
        "training_time_sec": training_time_sec,
        "peak_gpu_memory_mb": peak_gpu_memory_mb,
        "inference_time_sec": inference_time_sec,
        "inference_ms_per_recording":
            inference_ms_per_recording,

        "pretraining_status": "none; random initialization",
        "test_used_for_model_selection": False,
    }

    del model
    gc.collect()

    if DEVICE == "cuda":
        torch.cuda.empty_cache()

    return result


# ============================================================
# 16. Optional paired bootstrap AUC comparison:
# full 16-channel GAF vs raw 16-channel VGRF
#
# Requires:
# ./gait_gaf_ablation/full_16ch/
#     independent_test_subject_predictions.csv
# ============================================================
def paired_auc_difference_vs_gaf(raw_subject_df):
    gaf_path = os.path.join(
        GAF_ABLATION_ROOT,
        "full_16ch",
        "independent_test_subject_predictions.csv",
    )

    if not os.path.exists(gaf_path):
        print(
            "\nFull 16-channel GAF subject prediction file not found. "
            "Skipping paired raw-vs-GAF AUC comparison."
        )
        return None

    gaf = pd.read_csv(gaf_path)[
        ["subject_id", "y_true", "prob_PD"]
    ].rename(columns={"prob_PD": "prob_gaf"})

    raw = raw_subject_df[
        ["subject_id", "y_true", "prob_PD"]
    ].rename(columns={"prob_PD": "prob_raw"})

    merged = gaf.merge(
        raw,
        on=["subject_id", "y_true"],
        how="inner",
        validate="one_to_one",
    )

    if len(merged) != len(raw_subject_df):
        raise RuntimeError(
            "Raw and GAF predictions do not contain exactly the same "
            "independent-test subjects."
        )

    auc_gaf = roc_auc_score(
        merged["y_true"],
        merged["prob_gaf"],
    )
    auc_raw = roc_auc_score(
        merged["y_true"],
        merged["prob_raw"],
    )

    observed_delta = auc_gaf - auc_raw

    control = merged[
        merged["y_true"] == 0
    ].reset_index(drop=True)

    pd_group = merged[
        merged["y_true"] == 1
    ].reset_index(drop=True)

    rng = np.random.default_rng(SEED)
    deltas = []

    for _ in range(BOOTSTRAP_ITERATIONS):
        b0 = control.iloc[
            rng.integers(0, len(control), len(control))
        ]
        b1 = pd_group.iloc[
            rng.integers(0, len(pd_group), len(pd_group))
        ]

        boot = pd.concat([b0, b1], ignore_index=True)

        b_auc_gaf = roc_auc_score(
            boot["y_true"],
            boot["prob_gaf"],
        )
        b_auc_raw = roc_auc_score(
            boot["y_true"],
            boot["prob_raw"],
        )

        deltas.append(b_auc_gaf - b_auc_raw)

    deltas = np.asarray(deltas)

    lower_q = 100 * BOOTSTRAP_ALPHA / 2
    upper_q = 100 * (1 - BOOTSTRAP_ALPHA / 2)

    delta_low = float(np.percentile(deltas, lower_q))
    delta_high = float(np.percentile(deltas, upper_q))

    # Simple two-sided paired bootstrap sign probability.
    # Treat as an exploratory paired bootstrap comparison.
    p_left = np.mean(deltas <= 0)
    p_right = np.mean(deltas >= 0)
    p_boot = float(min(1.0, 2.0 * min(p_left, p_right)))

    merged.to_csv(
        os.path.join(
            OUTPUT_ROOT,
            "paired_raw_vs_gaf_subject_predictions.csv",
        ),
        index=False,
    )

    result = {
        "comparison":
            "full_16ch_GAF_SwinTiny_vs_raw_16ch_VGRF_1DCNN",
        "n_subjects": len(merged),
        "gaf_auc": auc_gaf,
        "raw_auc": auc_raw,
        "delta_auc_gaf_minus_raw": observed_delta,
        "delta_auc_ci_low": delta_low,
        "delta_auc_ci_high": delta_high,
        "paired_bootstrap_p": p_boot,
        "bootstrap_iterations": BOOTSTRAP_ITERATIONS,
    }

    pd.DataFrame([result]).to_csv(
        os.path.join(
            OUTPUT_ROOT,
            "paired_auc_raw_vs_gaf.csv",
        ),
        index=False,
    )

    return result


# ============================================================
# 17. Save environment / protocol
# ============================================================
def save_protocol():
    info = {
        "representation": "raw 16-channel VGRF time series",
        "model": "conventional 1D-CNN",
        "input_shape": [N_CHANNELS, SIGNAL_LENGTH],
        "gaf_used": False,
        "preprocessing": (
            "Each sensor-specific VGRF signal was independently "
            "resampled to 1024 points by linear interpolation and "
            "then independently standardized using z-score normalization."
        ),
        "seed": SEED,
        "test_size": TEST_SIZE,
        "n_folds": N_FOLDS,
        "reuse_gaf_subject_split": REUSE_GAF_SPLIT,
        "optimizer": "AdamW",
        "learning_rate": LR,
        "weight_decay": WEIGHT_DECAY,
        "maximum_epochs": EPOCHS,
        "early_stopping_patience": PATIENCE,
        "label_smoothing": LABEL_SMOOTHING,
        "batch_size": BATCH_SIZE,
        "gradient_accumulation_steps": GRAD_ACCUM_STEPS,
        "model_selection":
            "highest subject-level validation AUC in each fold",
        "final_epoch_rule":
            "median of fold-specific best epochs, rounded",
        "subject_aggregation":
            "mean probability across all recordings from the same subject",
        "classification_threshold": SUBJECT_THRESHOLD,
        "bootstrap_iterations": BOOTSTRAP_ITERATIONS,
        "bootstrap_type":
            "stratified subject-level percentile bootstrap",
        "pretraining": "none; random initialization",
        "device": DEVICE,
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
    }

    with open(
        os.path.join(OUTPUT_ROOT, "raw_vgrf_protocol.json"),
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(info, f, indent=2, ensure_ascii=False)


# ============================================================
# 18. Main
# ============================================================
def main():
    os.makedirs(OUTPUT_ROOT, exist_ok=True)
    set_seed(SEED)

    print("=" * 80)
    print("RAW VGRF NON-GAF BASELINE")
    print("=" * 80)
    print("Device:", DEVICE)
    print("Input:", f"{N_CHANNELS} x {SIGNAL_LENGTH}")
    print("Model: 1D-CNN")
    print("GAF used: No")

    meta = create_raw_dataset()

    dev_subjects, test_subjects = load_or_create_subject_split(meta)
    folds = make_folds(dev_subjects)

    development_df = meta[
        meta["subject_id"].isin(dev_subjects["subject_id"])
    ].copy()

    test_df = meta[
        meta["subject_id"].isin(test_subjects["subject_id"])
    ].copy()

    # Leakage safety check.
    dev_ids = set(development_df["subject_id"])
    test_ids = set(test_df["subject_id"])

    if dev_ids & test_ids:
        raise RuntimeError(
            "Subject leakage detected between development and test cohorts."
        )

    print(
        f"\nSubjects: total={meta['subject_id'].nunique()}, "
        f"development={development_df['subject_id'].nunique()}, "
        f"independent test={test_df['subject_id'].nunique()}"
    )

    print(
        f"Recordings: development={len(development_df)}, "
        f"independent test={len(test_df)}"
    )

    cv_results = []
    epoch_logs = []

    for fold_info in folds:
        fold = fold_info["fold"]

        train_df = development_df[
            development_df["subject_id"].isin(
                fold_info["train_subject_ids"]
            )
        ].copy()

        val_df = development_df[
            development_df["subject_id"].isin(
                fold_info["val_subject_ids"]
            )
        ].copy()

        fold_dir = os.path.join(
            OUTPUT_ROOT,
            f"cv_fold_{fold}",
        )
        os.makedirs(fold_dir, exist_ok=True)

        result, logs = train_cv_fold(
            fold,
            train_df,
            val_df,
            fold_dir,
        )

        cv_results.append(result)
        epoch_logs.extend(logs)

    best_epochs = [
        x["best_epoch"]
        for x in cv_results
    ]

    final_epochs = max(
        1,
        int(np.rint(np.median(best_epochs))),
    )

    final_result = train_final_and_test(
        development_df,
        test_df,
        final_epochs,
    )

    fold_aucs = [
        x["best_validation_subject_auc"]
        for x in cv_results
    ]

    final_result["cv_best_epochs"] = json.dumps(best_epochs)
    final_result["cv_mean_best_subject_auc"] = float(
        np.mean(fold_aucs)
    )
    final_result["cv_sd_best_subject_auc"] = float(
        np.std(fold_aucs, ddof=1)
    )

    pd.DataFrame(cv_results).to_csv(
        os.path.join(
            OUTPUT_ROOT,
            "raw_vgrf_5fold_results.csv",
        ),
        index=False,
    )

    pd.DataFrame(epoch_logs).to_csv(
        os.path.join(
            OUTPUT_ROOT,
            "raw_vgrf_epoch_logs.csv",
        ),
        index=False,
    )

    pd.DataFrame([final_result]).to_csv(
        os.path.join(
            OUTPUT_ROOT,
            "raw_vgrf_independent_test_results.csv",
        ),
        index=False,
    )

    # Manuscript-friendly output table.
    manuscript_cols = [
        "representation",
        "model",
        "input_channels",
        "signal_length",
        "subject_test_accuracy",
        "subject_test_precision",
        "subject_test_sensitivity",
        "subject_test_specificity",
        "subject_test_f1",
        "subject_test_auc",
        "auc_ci_low",
        "auc_ci_high",
        "total_parameters",
        "training_time_sec",
        "peak_gpu_memory_mb",
        "inference_ms_per_recording",
    ]

    pd.DataFrame([final_result])[
        manuscript_cols
    ].to_csv(
        os.path.join(
            OUTPUT_ROOT,
            "Table_raw_vgrf_baseline.csv",
        ),
        index=False,
    )

    raw_subject_df = pd.read_csv(
        os.path.join(
            OUTPUT_ROOT,
            "independent_test_subject_predictions.csv",
        )
    )

    paired_result = paired_auc_difference_vs_gaf(
        raw_subject_df
    )

    save_protocol()

    print("\n" + "=" * 80)
    print("FINAL RAW VGRF BASELINE RESULT")
    print("=" * 80)

    print(
        pd.DataFrame([final_result])[
            [
                "representation",
                "model",
                "final_training_epochs",
                "subject_test_accuracy",
                "subject_test_precision",
                "subject_test_sensitivity",
                "subject_test_specificity",
                "subject_test_f1",
                "subject_test_auc",
                "auc_ci_low",
                "auc_ci_high",
                "cv_mean_best_subject_auc",
                "cv_sd_best_subject_auc",
            ]
        ].to_string(index=False)
    )

    if paired_result is not None:
        print("\nPaired comparison with full 16-channel GAF:")
        print(
            pd.DataFrame([paired_result]).to_string(index=False)
        )

    print("\nOutputs saved to:", OUTPUT_ROOT)


if __name__ == "__main__":
    main()
