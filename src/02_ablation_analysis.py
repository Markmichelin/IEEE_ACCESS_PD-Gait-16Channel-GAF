import os
import re
import glob
import gc
import json
import time
import random
import shutil
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
from scipy.stats import binomtest
from statsmodels.stats.multitest import multipletests
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import timm
from pyts.image import GramianAngularField
DATA_ROOT = r"./gait-in-parkinsons-disease-1.0.0"
OUTPUT_ROOT = r"./gait_gaf_ablation"
SIGNAL_LENGTH = 1024
MODEL_NAME = "swin_tiny_patch4_window7_224"
IMAGE_SIZE = 1024

TEST_SIZE = 0.20
N_FOLDS = 5
SEED = 40

EPOCHS = 80
PATIENCE = 15
LR = 5e-6
WEIGHT_DECAY = 1e-2
LABEL_SMOOTHING = 0.05
NUM_WORKERS = 0
BATCH_SIZE = 1
GRAD_ACCUM_STEPS = 16

PRETRAINED = True
REQUIRE_PRETRAINED = True
RECREATE_ARRAYS = True

SUBJECT_THRESHOLD = 0.50
BOOTSTRAP_ITERATIONS = 2000
BOOTSTRAP_ALPHA = 0.05

ABLATIONS = [
    "full_16ch",
    "left_8ch",
    "right_8ch",
    "bilateral_mean_8ch",
    "global_mean_1ch",
]


def select_device():
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


DEVICE = select_device()
def set_seed(seed=40):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

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

    return data[:, 1:17].astype(np.float32)

def preprocess_16_sensors(sensors, length=1024):
    if sensors is None or sensors.shape[0] < 10 or sensors.shape[1] != 16:
        return None

    old_x = np.linspace(0, 1, sensors.shape[0])
    new_x = np.linspace(0, 1, length)

    processed = []

    for i in range(16):
        sig = sensors[:, i].astype(np.float32)

        sig = np.interp(new_x, old_x, sig)

        sig = (
            sig - np.mean(sig)
        ) / (
            np.std(sig) + 1e-8
        )

        processed.append(sig)

    return np.stack(processed, axis=0).astype(np.float32)

def signals_to_16channel_gaf(signals_16, image_size):
    gaf = GramianAngularField(
        image_size=image_size,
        method="summation",
    )

    gaf_images = gaf.fit_transform(signals_16).astype(np.float32)

    normalized = []

    for x in gaf_images:
        x = x - x.min()
        x = x / (x.max() + 1e-8)
        normalized.append(x)

    return np.stack(normalized, axis=0).astype(np.float32)

def make_ablation_array(gaf16, ablation_name):
    if gaf16.shape[0] != 16:
        raise ValueError(f"Expected 16 GAF channels, got {gaf16.shape}")

    if ablation_name == "full_16ch":
        out = gaf16

    elif ablation_name == "left_8ch":
        out = gaf16[:8]

    elif ablation_name == "right_8ch":
        out = gaf16[8:16]

    elif ablation_name == "bilateral_mean_8ch":
        out = (gaf16[:8] + gaf16[8:16]) / 2.0

    elif ablation_name == "global_mean_1ch":
        out = np.mean(gaf16, axis=0, keepdims=True)

    else:
        raise ValueError(f"Unknown ablation: {ablation_name}")

    return out.astype(np.float16)


def ablation_in_chans(ablation_name):
    return {
        "full_16ch": 16,
        "left_8ch": 8,
        "right_8ch": 8,
        "bilateral_mean_8ch": 8,
        "global_mean_1ch": 1,
    }[ablation_name]

def create_ablation_datasets():
    if RECREATE_ARRAYS and os.path.exists(OUTPUT_ROOT):
        shutil.rmtree(OUTPUT_ROOT)

    os.makedirs(OUTPUT_ROOT, exist_ok=True)

    array_dirs = {}

    for ablation in ABLATIONS:
        d = os.path.join(OUTPUT_ROOT, ablation, "arrays")
        os.makedirs(d, exist_ok=True)
        array_dirs[ablation] = d

    files = sorted(
        glob.glob(
            os.path.join(DATA_ROOT, "**", "*.txt"),
            recursive=True,
        )
    )

    rows = {a: [] for a in ABLATIONS}
    exclusions = []

    for file_path in tqdm(files, desc="Creating ablation GAF arrays"):
        label = get_label_from_filename(file_path)

        if label is None:
            continue

        try:
            sensors = read_16_sensors(file_path)
            signals16 = preprocess_16_sensors(sensors, SIGNAL_LENGTH)

            if signals16 is None:
                exclusions.append({
                    "source_file": file_path,
                    "reason": "invalid/short 16-sensor signal",
                })
                continue

            gaf16 = signals_to_16channel_gaf(
                signals16,
                IMAGE_SIZE,
            )

            base = os.path.splitext(os.path.basename(file_path))[0]
            subject_id = get_subject_id(file_path)

            for ablation in ABLATIONS:
                arr = make_ablation_array(gaf16, ablation)

                save_path = os.path.join(
                    array_dirs[ablation],
                    base + ".npy",
                )

                np.save(save_path, arr)

                rows[ablation].append({
                    "array_path": save_path,
                    "label": label,
                    "label_name": "PD" if label == 1 else "Control",
                    "subject_id": subject_id,
                    "source_file": file_path,
                    "ablation": ablation,
                    "in_chans": arr.shape[0],
                    "image_size": IMAGE_SIZE,
                    "shape": str(arr.shape),
                })

        except Exception as e:
            exclusions.append({
                "source_file": file_path,
                "reason": str(e),
            })

    metas = {}

    for ablation in ABLATIONS:
        meta = pd.DataFrame(rows[ablation])

        if meta.empty:
            raise ValueError(f"No arrays created for {ablation}")

        meta.to_csv(
            os.path.join(OUTPUT_ROOT, ablation, "metadata.csv"),
            index=False,
        )

        metas[ablation] = meta

    pd.DataFrame(
        exclusions,
        columns=["source_file", "reason"],
    ).to_csv(
        os.path.join(OUTPUT_ROOT, "exclusions.csv"),
        index=False,
    )

    return metas

def make_subject_split(reference_meta):
    subject_df = (
        reference_meta[["subject_id", "label"]]
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

    for fold, (train_idx, val_idx) in enumerate(
        skf.split(X, y),
        start=1,
    ):
        folds.append({
            "fold": fold,
            "train_subject_ids":
                set(dev_subjects.iloc[train_idx]["subject_id"]),
            "val_subject_ids":
                set(dev_subjects.iloc[val_idx]["subject_id"]),
        })

    return folds

class GAFDataset(Dataset):
    def __init__(self, df):
        self.df = df.reset_index(drop=True)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        arr = np.load(
            self.df.loc[idx, "array_path"]
        ).astype(np.float32)

        label = int(self.df.loc[idx, "label"])

        return (
            torch.from_numpy(arr),
            torch.tensor(label, dtype=torch.long),
        )


def make_loader(df, shuffle):
    return DataLoader(
        GAFDataset(df),
        batch_size=BATCH_SIZE,
        shuffle=shuffle,
        num_workers=NUM_WORKERS,
    )

def build_model(ablation_name):
    in_chans = ablation_in_chans(ablation_name)

    kwargs = {
        "pretrained": PRETRAINED,
        "num_classes": 2,
        "in_chans": in_chans,
        "img_size": IMAGE_SIZE,
    }

    try:
        model = timm.create_model(
            MODEL_NAME,
            **kwargs,
        )

        init_status = (
            f"ImageNet-pretrained; timm in_chans={in_chans} adaptation"
        )

    except Exception as e:
        if REQUIRE_PRETRAINED:
            raise RuntimeError(
                f"Pretrained model construction failed for "
                f"{ablation_name}, in_chans={in_chans}: {e}"
            )

        kwargs["pretrained"] = False
        model = timm.create_model(MODEL_NAME, **kwargs)
        init_status = "random initialization"

    return model.to(DEVICE), init_status


def count_parameters(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )
    return total, trainable

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
    y_true = []
    y_pred = []
    y_prob = []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            outputs = model(images)
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

def bootstrap_subject_metrics(subject_df):
    rng = np.random.default_rng(SEED)

    control = (
        subject_df[subject_df["y_true"] == 0]
        .reset_index(drop=True)
    )

    pd_group = (
        subject_df[subject_df["y_true"] == 1]
        .reset_index(drop=True)
    )

    metric_names = [
        "accuracy",
        "precision",
        "sensitivity",
        "specificity",
        "f1",
        "auc",
    ]

    values = {k: [] for k in metric_names}

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

        for key in metric_names:
            if not np.isnan(m[key]):
                values[key].append(m[key])

    lo = 100 * BOOTSTRAP_ALPHA / 2
    hi = 100 * (1 - BOOTSTRAP_ALPHA / 2)

    result = {}

    for key, vals in values.items():
        result[f"{key}_ci_low"] = float(np.percentile(vals, lo))
        result[f"{key}_ci_high"] = float(np.percentile(vals, hi))

    return result

def make_criterion(train_df):
    counts = train_df["label"].value_counts().sort_index()

    weights = np.asarray(
        [
            1.0 / max(counts.get(0, 1), 1),
            1.0 / max(counts.get(1, 1), 1),
        ],
        dtype=np.float32,
    )

    weights = weights / weights.sum() * 2

    return nn.CrossEntropyLoss(
        weight=torch.tensor(
            weights,
            dtype=torch.float32,
            device=DEVICE,
        ),
        label_smoothing=LABEL_SMOOTHING,
    )

def train_one_epoch(
    model,
    loader,
    criterion,
    optimizer,
):
    model.train()
    total_loss = 0.0

    optimizer.zero_grad(set_to_none=True)

    for step, (images, labels) in enumerate(loader, start=1):
        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        outputs = model(images)

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

def train_cv_fold(
    ablation_name,
    fold_number,
    train_df,
    val_df,
    fold_dir,
):
    set_seed(SEED + fold_number)

    train_loader = make_loader(train_df, True)
    val_loader = make_loader(val_df, False)

    criterion = make_criterion(train_df)

    model, init_status = build_model(ablation_name)

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
            "ablation": ablation_name,
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
            f"{ablation_name} | Fold {fold_number} | "
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

    fold_result = {
        "ablation": ablation_name,
        "fold": fold_number,
        "input_channels": ablation_in_chans(ablation_name),
        "train_subjects": train_df["subject_id"].nunique(),
        "validation_subjects": val_df["subject_id"].nunique(),
        "train_recordings": len(train_df),
        "validation_recordings": len(val_df),
        "best_epoch": best_epoch,
        "best_validation_subject_auc": best_auc,
        "pretraining_status": init_status,
    }

    del model
    gc.collect()

    if DEVICE == "cuda":
        torch.cuda.empty_cache()

    return fold_result, epoch_logs

def train_final_and_test(
    ablation_name,
    development_df,
    test_df,
    final_epochs,
    output_dir,
):
    print("\n" + "=" * 80)
    print(
        f"FINAL: {ablation_name} | {MODEL_NAME} | "
        f"{IMAGE_SIZE}x{IMAGE_SIZE} | epochs={final_epochs}"
    )
    print("=" * 80)

    set_seed(SEED)

    train_loader = make_loader(development_df, True)
    test_loader = make_loader(test_df, False)

    criterion = make_criterion(development_df)

    model, init_status = build_model(ablation_name)

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
            f"{ablation_name} | Final epoch "
            f"{epoch:03d}/{final_epochs} | Loss={loss:.4f}"
        )

    training_time_sec = time.time() - start

    peak_gpu_memory_mb = np.nan

    if DEVICE == "cuda":
        peak_gpu_memory_mb = (
            torch.cuda.max_memory_allocated() / (1024 ** 2)
        )

    if DEVICE == "cuda":
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
        1000 * inference_time_sec / max(len(test_df), 1)
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
            output_dir,
            "independent_test_recording_predictions.csv",
        ),
        index=False,
    )

    subject_df.to_csv(
        os.path.join(
            output_dir,
            "independent_test_subject_predictions.csv",
        ),
        index=False,
    )

    torch.save(
        model.state_dict(),
        os.path.join(output_dir, "final_model.pth"),
    )

    result = {
        "ablation": ablation_name,
        "model": MODEL_NAME,
        "image_size": IMAGE_SIZE,
        "input_channels": ablation_in_chans(ablation_name),
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

        "subject_confusion_matrix":
            json.dumps(subject_result["confusion_matrix"].tolist()),

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

        "pretraining_status": init_status,
        "test_used_for_model_selection": False,
    }

    del model
    gc.collect()

    if DEVICE == "cuda":
        torch.cuda.empty_cache()

    return result

def exact_mcnemar(y_true, pred_a, pred_b):
    y_true = np.asarray(y_true)
    a = np.asarray(pred_a)
    b = np.asarray(pred_b)

    a_correct = (a == y_true)
    b_correct = (b == y_true)

    n01 = int(np.sum(a_correct & ~b_correct))
    n10 = int(np.sum(~a_correct & b_correct))

    n = n01 + n10

    if n == 0:
        return n01, n10, 1.0

    p = binomtest(
        min(n01, n10),
        n=n,
        p=0.5,
        alternative="two-sided",
    ).pvalue

    return n01, n10, p


def compare_ablation_predictions():
    reference_name = "full_16ch"

    ref_path = os.path.join(
        OUTPUT_ROOT,
        reference_name,
        "independent_test_subject_predictions.csv",
    )

    ref = pd.read_csv(ref_path)[
        ["subject_id", "y_true", "y_pred"]
    ].rename(columns={"y_pred": "pred_full"})

    rows = []

    for ablation in ABLATIONS:
        if ablation == reference_name:
            continue

        other = pd.read_csv(
            os.path.join(
                OUTPUT_ROOT,
                ablation,
                "independent_test_subject_predictions.csv",
            )
        )[["subject_id", "y_true", "y_pred"]].rename(
            columns={"y_pred": "pred_other"}
        )

        merged = ref.merge(
            other,
            on=["subject_id", "y_true"],
            how="inner",
            validate="one_to_one",
        )

        n01, n10, p = exact_mcnemar(
            merged["y_true"],
            merged["pred_full"],
            merged["pred_other"],
        )

        rows.append({
            "comparison":
                f"full_16ch vs {ablation}",
            "n_subjects": len(merged),
            "full_correct_other_wrong": n01,
            "full_wrong_other_correct": n10,
            "mcnemar_exact_p": p,
        })

    stats_df = pd.DataFrame(rows)

    reject, p_holm, _, _ = multipletests(
        stats_df["mcnemar_exact_p"].values,
        alpha=0.05,
        method="holm",
    )

    stats_df["holm_adjusted_p"] = p_holm
    stats_df["significant_after_holm"] = reject

    stats_df.to_csv(
        os.path.join(
            OUTPUT_ROOT,
            "ablation_mcnemar_holm.csv",
        ),
        index=False,
    )

    return stats_df

def main():
    set_seed(SEED)
    print("Device:", DEVICE)
    print("Model:", MODEL_NAME)
    print("Image size:", IMAGE_SIZE)
    print("Ablations:", ABLATIONS)
    metas = create_ablation_datasets()
    reference_sources = set(metas["full_16ch"]["source_file"])
    for ablation in ABLATIONS:
        current_sources = set(metas[ablation]["source_file"])

        if current_sources != reference_sources:
            raise RuntimeError(
                f"Recording mismatch in {ablation}. "
                "All ablations must use identical recordings."
            )

    reference_meta = metas["full_16ch"]

    dev_subjects, test_subjects = make_subject_split(reference_meta)
    folds = make_folds(dev_subjects)

    print(
        f"\nSubjects: total={reference_meta['subject_id'].nunique()}, "
        f"development={len(dev_subjects)}, "
        f"independent_test={len(test_subjects)}"
    )

    all_cv_results = []
    all_epoch_logs = []
    final_results = []

    for ablation in ABLATIONS:
        print("\n" + "#" * 80)
        print("ABLATION:", ablation)
        print("#" * 80)

        meta = metas[ablation]

        development_df = meta[
            meta["subject_id"].isin(dev_subjects["subject_id"])
        ].copy()

        test_df = meta[
            meta["subject_id"].isin(test_subjects["subject_id"])
        ].copy()

        ablation_dir = os.path.join(OUTPUT_ROOT, ablation)
        os.makedirs(ablation_dir, exist_ok=True)

        fold_results = []

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
                ablation_dir,
                f"cv_fold_{fold}",
            )

            os.makedirs(fold_dir, exist_ok=True)

            result, logs = train_cv_fold(
                ablation,
                fold,
                train_df,
                val_df,
                fold_dir,
            )

            fold_results.append(result)
            all_cv_results.append(result)
            all_epoch_logs.extend(logs)

        best_epochs = [
            x["best_epoch"]
            for x in fold_results
        ]

        final_epochs = max(
            1,
            int(np.rint(np.median(best_epochs))),
        )

        final_result = train_final_and_test(
            ablation,
            development_df,
            test_df,
            final_epochs,
            ablation_dir,
        )

        final_result["cv_best_epochs"] = json.dumps(best_epochs)
        final_result["cv_mean_best_subject_auc"] = float(
            np.mean([
                x["best_validation_subject_auc"]
                for x in fold_results
            ])
        )
        final_result["cv_sd_best_subject_auc"] = float(
            np.std(
                [
                    x["best_validation_subject_auc"]
                    for x in fold_results
                ],
                ddof=1,
            )
        )

        final_results.append(final_result)

    cv_df = pd.DataFrame(all_cv_results)
    cv_df.to_csv(
        os.path.join(OUTPUT_ROOT, "ablation_5fold_results.csv"),
        index=False,
    )

    pd.DataFrame(all_epoch_logs).to_csv(
        os.path.join(OUTPUT_ROOT, "ablation_epoch_logs.csv"),
        index=False,
    )

    final_df = pd.DataFrame(final_results)
    final_df.to_csv(
        os.path.join(
            OUTPUT_ROOT,
            "ablation_independent_test_results.csv",
        ),
        index=False,
    )

    stats_df = compare_ablation_predictions()
    manuscript = final_df[
        [
            "ablation",
            "input_channels",
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
    ].copy()

    manuscript.to_csv(
        os.path.join(
            OUTPUT_ROOT,
            "Table_ablation_manuscript.csv",
        ),
        index=False,
    )

    protocol = {
        "model": MODEL_NAME,
        "image_size": IMAGE_SIZE,
        "signal_length": SIGNAL_LENGTH,
        "ablations": ABLATIONS,
        "seed": SEED,
        "test_size": TEST_SIZE,
        "n_folds": N_FOLDS,
        "optimizer": "AdamW",
        "learning_rate": LR,
        "weight_decay": WEIGHT_DECAY,
        "maximum_epochs": EPOCHS,
        "patience": PATIENCE,
        "label_smoothing": LABEL_SMOOTHING,
        "batch_size": BATCH_SIZE,
        "gradient_accumulation_steps": GRAD_ACCUM_STEPS,
        "pretrained": PRETRAINED,
        "model_selection":
            "highest subject-level validation AUC in each development fold",
        "final_epoch_rule":
            "median of five fold-specific best epochs, rounded",
        "subject_aggregation":
            "mean probability across recordings",
        "subject_threshold": SUBJECT_THRESHOLD,
        "bootstrap":
            "2000 stratified subject-level percentile bootstrap resamples",
        "statistical_comparison":
            "exact McNemar tests comparing full_16ch with each ablation; Holm correction",
        "device": DEVICE,
        "python": platform.python_version(),
        "torch": torch.__version__,
        "timm": getattr(timm, "__version__", "unknown"),
    }

    with open(
        os.path.join(OUTPUT_ROOT, "ablation_protocol.json"),
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(protocol, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 80)
    print("FINAL ABLATION RESULTS")
    print("=" * 80)
    print(
        final_df[
            [
                "ablation",
                "input_channels",
                "final_training_epochs",
                "subject_test_accuracy",
                "subject_test_sensitivity",
                "subject_test_specificity",
                "subject_test_f1",
                "subject_test_auc",
                "auc_ci_low",
                "auc_ci_high",
            ]
        ].to_string(index=False)
    )

    print("\nMcNemar + Holm:")
    print(stats_df.to_string(index=False))

    print("\nSaved to:", OUTPUT_ROOT)


if __name__ == "__main__":
    main()
