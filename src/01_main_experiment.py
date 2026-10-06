
import os
import re
import glob
import random
import shutil
import time
import json
import platform
import itertools
import math
import gc
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


# =========================================================
# 1. Settings
# =========================================================
DATA_ROOT = r"./gait-in-parkinsons-disease-1.0.0"
OUTPUT_ROOT = r"./gait_16channel_gaf_5fold_with_independent_test"

IMAGE_SIZES = [64, 128, 224, 512, 1024]
SIGNAL_LENGTH = 1024

# Independent holdout test set
TEST_SIZE = 0.20

# 5-fold CV is performed ONLY on the remaining development subjects
N_FOLDS = 5
SEED = 40

EPOCHS = 80
PATIENCE = 15
LR = 5e-6
WEIGHT_DECAY = 1e-2
NUM_WORKERS = 0
LABEL_SMOOTHING = 0.05

BATCH_SIZE_MAP = {
    64: 16,
    128: 8,
    224: 4,
    512: 1,
    1024: 1,
}

GRAD_ACCUM_MAP = {
    64: 1,
    128: 2,
    224: 4,
    512: 16,
    1024: 16,
}

RECREATE_ARRAYS = True
PRETRAINED = True
REQUIRE_PRETRAINED = True

BOOTSTRAP_ITERATIONS = 2000
BOOTSTRAP_ALPHA = 0.05
BOOTSTRAP_STRATIFIED = True
SUBJECT_THRESHOLD = 0.50

MODEL_LIST = [
    "resnet50",
    "efficientnet_b0",
    "convnext_tiny",
    "vit_tiny_patch16_224",
    "swin_tiny_patch4_window7_224",
]


def select_device():
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


DEVICE = select_device()


# =========================================================
# 2. Reproducibility / environment
# =========================================================
def set_seed(seed=40):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def save_environment_info():
    info = {
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "torch_version": torch.__version__,
        "timm_version": getattr(timm, "__version__", "unknown"),
        "numpy_version": np.__version__,
        "pandas_version": pd.__version__,
        "device": DEVICE,
        "seed": SEED,
        "test_size": TEST_SIZE,
        "n_folds": N_FOLDS,
        "epochs": EPOCHS,
        "patience": PATIENCE,
        "learning_rate": LR,
        "weight_decay": WEIGHT_DECAY,
        "label_smoothing": LABEL_SMOOTHING,
        "signal_length": SIGNAL_LENGTH,
        "image_sizes": IMAGE_SIZES,
        "models": MODEL_LIST,
        "pretrained": PRETRAINED,
        "require_pretrained": REQUIRE_PRETRAINED,
        "bootstrap_iterations": BOOTSTRAP_ITERATIONS,
        "bootstrap_stratified": BOOTSTRAP_STRATIFIED,
        "bootstrap_ci_type": "percentile",
        "subject_probability_aggregation": "mean recording probability",
        "subject_threshold": SUBJECT_THRESHOLD,
        "split_method": (
            "stratified subject-level 20% independent holdout test set; "
            "subject-wise stratified 5-fold cross-validation on the remaining 80% development subjects"
        ),
        "cv_role": (
            "5-fold CV is used only for development/validation and epoch robustness; "
            "the independent holdout test set is never used during CV or model selection"
        ),
        "final_epoch_rule": (
            "median of best epochs across the five development folds, rounded to nearest integer"
        ),
    }

    if torch.cuda.is_available():
        info["gpu_name"] = torch.cuda.get_device_name(0)
    elif DEVICE == "mps":
        info["gpu_name"] = "Apple Metal Performance Shaders (MPS)"
    else:
        info["gpu_name"] = "CPU"

    os.makedirs(OUTPUT_ROOT, exist_ok=True)

    with open(
        os.path.join(OUTPUT_ROOT, "environment_and_protocol.json"),
        "w",
        encoding="utf-8"
    ) as f:
        json.dump(info, f, indent=2, ensure_ascii=False)


# =========================================================
# 3. Label / subject / study parsing
# =========================================================
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


def get_study_id(filename):
    name = os.path.basename(filename)

    if name.startswith("Ga"):
        return "Ga"
    if name.startswith("Ju"):
        return "Ju"
    if name.startswith("Si"):
        return "Si"

    return "Unknown"


def get_walk_number(filename):
    stem = os.path.splitext(os.path.basename(filename))[0]
    parts = stem.split("_")
    return parts[-1] if len(parts) > 1 else ""


# =========================================================
# 4. Read gait file
# time + L1-L8 + R1-R8 + left/right total force
# Only L1-L8 + R1-R8 are used.
# =========================================================
def read_16_sensors(file_path):
    df = pd.read_csv(
        file_path,
        sep=r"\s+",
        header=None,
        engine="python"
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


# =========================================================
# 5. Preprocess to common 1024-point sequence
# =========================================================
def preprocess_16_sensors(sensors, length=1024):
    if sensors is None or sensors.shape[0] < 10 or sensors.shape[1] != 16:
        return None

    processed = []

    old_x = np.linspace(0, 1, sensors.shape[0])
    new_x = np.linspace(0, 1, length)

    for i in range(16):
        sig = sensors[:, i].astype(np.float32)

        sig = np.interp(
            new_x,
            old_x,
            sig
        )

        sig = (
            sig - np.mean(sig)
        ) / (
            np.std(sig) + 1e-8
        )

        processed.append(sig)

    return np.stack(
        processed,
        axis=0
    ).astype(np.float32)


# =========================================================
# 6. Direct GAF generation
# No post-hoc bilinear resizing.
# =========================================================
def signals_to_16channel_gaf(signals_16, image_size):
    if image_size > SIGNAL_LENGTH:
        raise ValueError(
            "GAF image_size cannot exceed SIGNAL_LENGTH."
        )

    gaf = GramianAngularField(
        image_size=image_size,
        method="summation"
    )

    gaf_images = gaf.fit_transform(
        signals_16
    ).astype(np.float32)

    out = []

    for x in gaf_images:
        x = x - x.min()
        x = x / (x.max() + 1e-8)
        out.append(x)

    return np.stack(
        out,
        axis=0
    ).astype(np.float16)


# =========================================================
# 7. Build arrays / cohort audit
# =========================================================
def create_array_dataset(image_size):
    output_dir = os.path.join(
        OUTPUT_ROOT,
        f"size_{image_size}"
    )

    if RECREATE_ARRAYS and os.path.exists(output_dir):
        shutil.rmtree(output_dir)

    os.makedirs(output_dir, exist_ok=True)

    array_dir = os.path.join(
        output_dir,
        "arrays"
    )
    os.makedirs(array_dir, exist_ok=True)

    files = sorted(
        glob.glob(
            os.path.join(
                DATA_ROOT,
                "**",
                "*.txt"
            ),
            recursive=True
        )
    )

    records = []
    exclusions = []

    for file_path in tqdm(
        files,
        desc=f"Creating GAF {image_size}x{image_size}"
    ):
        label = get_label_from_filename(
            file_path
        )

        if label is None:
            exclusions.append({
                "source_file": file_path,
                "reason": "filename did not contain Co or Pt"
            })
            continue

        try:
            sensors = read_16_sensors(
                file_path
            )

            signals_16 = preprocess_16_sensors(
                sensors,
                SIGNAL_LENGTH
            )

            if signals_16 is None:
                exclusions.append({
                    "source_file": file_path,
                    "reason": "invalid/short 16-sensor signal"
                })
                continue

            arr = signals_to_16channel_gaf(
                signals_16,
                image_size
            )

            base = os.path.splitext(
                os.path.basename(file_path)
            )[0]

            save_path = os.path.join(
                array_dir,
                base + ".npy"
            )

            np.save(
                save_path,
                arr
            )

            records.append({
                "array_path": save_path,
                "label": label,
                "label_name": "PD" if label == 1 else "Control",
                "subject_id": get_subject_id(file_path),
                "study_id": get_study_id(file_path),
                "walk_number": get_walk_number(file_path),
                "source_file": file_path,
                "n_original_timepoints": sensors.shape[0],
                "image_size": image_size,
                "shape": str(arr.shape),
            })

        except Exception as e:
            exclusions.append({
                "source_file": file_path,
                "reason": str(e)
            })

    meta = pd.DataFrame(records)

    if meta.empty:
        raise ValueError(
            "No eligible recordings were created."
        )

    meta.to_csv(
        os.path.join(
            output_dir,
            "metadata.csv"
        ),
        index=False
    )

    pd.DataFrame(
        exclusions
    ).to_csv(
        os.path.join(
            output_dir,
            "exclusions.csv"
        ),
        index=False
    )

    recordings_per_subject = (
        meta.groupby(
            [
                "subject_id",
                "label",
                "label_name",
                "study_id"
            ],
            as_index=False
        )
        .agg(
            recording_count=(
                "source_file",
                "count"
            )
        )
    )

    recordings_per_subject.to_csv(
        os.path.join(
            output_dir,
            "recordings_per_subject.csv"
        ),
        index=False
    )

    cohort_summary = pd.DataFrame([{
        "image_size": image_size,
        "included_subjects":
            meta["subject_id"].nunique(),
        "included_recordings":
            len(meta),

        "control_subjects":
            meta.loc[
                meta.label == 0,
                "subject_id"
            ].nunique(),

        "pd_subjects":
            meta.loc[
                meta.label == 1,
                "subject_id"
            ].nunique(),

        "control_recordings":
            int((meta.label == 0).sum()),

        "pd_recordings":
            int((meta.label == 1).sum()),

        "excluded_files":
            len(exclusions),
    }])

    cohort_summary.to_csv(
        os.path.join(
            output_dir,
            "cohort_summary.csv"
        ),
        index=False
    )

    return meta, output_dir


# =========================================================
# 8. Independent TEST set + development subjects
# =========================================================
def make_independent_test_split(meta, output_dir):
    subject_df = (
        meta[
            ["subject_id", "label"]
        ]
        .drop_duplicates()
        .reset_index(drop=True)
    )

    development_subjects, test_subjects = train_test_split(
        subject_df,
        test_size=TEST_SIZE,
        random_state=SEED,
        stratify=subject_df["label"],
    )

    development_df = meta[
        meta["subject_id"].isin(
            development_subjects["subject_id"]
        )
    ].copy()

    test_df = meta[
        meta["subject_id"].isin(
            test_subjects["subject_id"]
        )
    ].copy()

    development_df.to_csv(
        os.path.join(
            output_dir,
            "development_all.csv"
        ),
        index=False
    )

    test_df.to_csv(
        os.path.join(
            output_dir,
            "independent_test.csv"
        ),
        index=False
    )

    rows = []

    for split_name, split_df in [
        ("development", development_df),
        ("independent_test", test_df),
    ]:
        rows.append({
            "split": split_name,
            "subjects_total":
                split_df["subject_id"].nunique(),

            "subjects_pd":
                split_df.loc[
                    split_df.label == 1,
                    "subject_id"
                ].nunique(),

            "subjects_control":
                split_df.loc[
                    split_df.label == 0,
                    "subject_id"
                ].nunique(),

            "recordings_total":
                len(split_df),

            "recordings_pd":
                int((split_df.label == 1).sum()),

            "recordings_control":
                int((split_df.label == 0).sum()),

            "random_seed":
                SEED,
        })

    split_summary = pd.DataFrame(
        rows
    )

    split_summary.to_csv(
        os.path.join(
            output_dir,
            "independent_test_split_summary.csv"
        ),
        index=False
    )

    return (
        development_subjects.reset_index(drop=True),
        test_subjects.reset_index(drop=True),
        development_df,
        test_df,
        split_summary
    )


# =========================================================
# 9. Build subject-wise 5-fold CV on DEVELOPMENT only
# =========================================================
def make_development_folds(
    development_subjects,
    development_df,
    output_dir
):
    skf = StratifiedKFold(
        n_splits=N_FOLDS,
        shuffle=True,
        random_state=SEED
    )

    X = development_subjects[
        "subject_id"
    ].values

    y = development_subjects[
        "label"
    ].values

    fold_data = []
    summary_rows = []

    for fold, (
        train_idx,
        val_idx
    ) in enumerate(
        skf.split(X, y),
        start=1
    ):
        fold_train_subjects = (
            development_subjects
            .iloc[train_idx]
            .copy()
        )

        fold_val_subjects = (
            development_subjects
            .iloc[val_idx]
            .copy()
        )

        train_df = development_df[
            development_df["subject_id"].isin(
                fold_train_subjects["subject_id"]
            )
        ].copy()

        val_df = development_df[
            development_df["subject_id"].isin(
                fold_val_subjects["subject_id"]
            )
        ].copy()

        fold_dir = os.path.join(
            output_dir,
            f"cv_fold_{fold}"
        )

        os.makedirs(
            fold_dir,
            exist_ok=True
        )

        train_df.to_csv(
            os.path.join(
                fold_dir,
                "train.csv"
            ),
            index=False
        )

        val_df.to_csv(
            os.path.join(
                fold_dir,
                "validation.csv"
            ),
            index=False
        )

        for split_name, split_df in [
            ("train", train_df),
            ("validation", val_df),
        ]:
            summary_rows.append({
                "fold":
                    fold,

                "split":
                    split_name,

                "subjects_total":
                    split_df[
                        "subject_id"
                    ].nunique(),

                "subjects_pd":
                    split_df.loc[
                        split_df.label == 1,
                        "subject_id"
                    ].nunique(),

                "subjects_control":
                    split_df.loc[
                        split_df.label == 0,
                        "subject_id"
                    ].nunique(),

                "recordings_total":
                    len(split_df),

                "recordings_pd":
                    int(
                        (split_df.label == 1).sum()
                    ),

                "recordings_control":
                    int(
                        (split_df.label == 0).sum()
                    ),
            })

        fold_data.append({
            "fold":
                fold,

            "fold_dir":
                fold_dir,

            "train_df":
                train_df,

            "val_df":
                val_df,
        })

    fold_summary = pd.DataFrame(
        summary_rows
    )

    fold_summary.to_csv(
        os.path.join(
            output_dir,
            "5fold_development_partition_summary.csv"
        ),
        index=False
    )

    return fold_data, fold_summary


# =========================================================
# 10. Torch dataset
# =========================================================
class Gait16ChannelGAFDataset(Dataset):
    def __init__(self, df):
        self.df = df.reset_index(
            drop=True
        )

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        arr = np.load(
            self.df.loc[
                idx,
                "array_path"
            ]
        ).astype(np.float32)

        label = int(
            self.df.loc[
                idx,
                "label"
            ]
        )

        return (
            torch.from_numpy(arr),
            torch.tensor(
                label,
                dtype=torch.long
            )
        )


# =========================================================
# 11. Model
# =========================================================
def build_model(
    model_name,
    image_size
):
    kwargs = {
        "pretrained":
            PRETRAINED,
        "num_classes":
            2,
        "in_chans":
            16,
    }

    if (
        "vit" in model_name
        or "swin" in model_name
    ):
        kwargs["img_size"] = (
            image_size
        )

    try:
        model = timm.create_model(
            model_name,
            **kwargs
        )

        init_status = (
            "ImageNet-pretrained; "
            "timm in_chans=16 adaptation"
        )

    except Exception as e:
        if REQUIRE_PRETRAINED:
            raise RuntimeError(
                f"Pretrained 16-channel "
                f"construction failed for "
                f"{model_name} at "
                f"{image_size}: {e}"
            )

        kwargs["pretrained"] = False

        model = timm.create_model(
            model_name,
            **kwargs
        )

        init_status = (
            "random initialization"
        )

    return (
        model.to(DEVICE),
        init_status
    )


def count_parameters(model):
    total = sum(
        p.numel()
        for p in model.parameters()
    )

    trainable = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )

    return total, trainable


# =========================================================
# 12. Metrics
# =========================================================
def calculate_metrics(
    y_true,
    y_pred,
    y_prob,
    loss=np.nan
):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    y_prob = np.asarray(y_prob)

    acc = accuracy_score(
        y_true,
        y_pred
    )

    precision, recall, f1, _ = (
        precision_recall_fscore_support(
            y_true,
            y_pred,
            average="binary",
            zero_division=0
        )
    )

    try:
        auc = roc_auc_score(
            y_true,
            y_prob
        )
    except Exception:
        auc = np.nan

    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=[0, 1]
    )

    tn, fp, fn, tp = cm.ravel()

    specificity = (
        tn / (tn + fp + 1e-8)
    )

    return {
        "loss":
            loss,
        "accuracy":
            acc,
        "precision":
            precision,
        "recall":
            recall,
        "specificity":
            specificity,
        "f1":
            f1,
        "auc":
            auc,
        "confusion_matrix":
            cm,
        "y_true":
            y_true.tolist(),
        "y_pred":
            y_pred.tolist(),
        "y_prob":
            y_prob.tolist(),
    }


def evaluate_recording_level(
    model,
    loader,
    criterion
):
    model.eval()

    total_loss = 0.0
    y_true = []
    y_pred = []
    y_prob = []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(
                DEVICE
            )

            labels = labels.to(
                DEVICE
            )

            outputs = model(images)

            loss = criterion(
                outputs,
                labels
            )

            probs = torch.softmax(
                outputs,
                dim=1
            )[:, 1]

            preds = torch.argmax(
                outputs,
                dim=1
            )

            total_loss += loss.item()

            y_true.extend(
                labels.cpu().numpy()
            )

            y_pred.extend(
                preds.cpu().numpy()
            )

            y_prob.extend(
                probs.cpu().numpy()
            )

    return calculate_metrics(
        y_true,
        y_pred,
        y_prob,
        total_loss / max(
            len(loader),
            1
        )
    )


# =========================================================
# 13. Subject aggregation
# =========================================================
def aggregate_to_subject_level(
    recording_df,
    threshold=0.50
):
    subject_df = (
        recording_df.groupby(
            [
                "subject_id",
                "label",
                "label_name"
            ],
            as_index=False
        )
        .agg(
            prob_PD=(
                "prob_PD",
                "mean"
            ),
            n_recordings=(
                "prob_PD",
                "size"
            ),
        )
    )

    subject_df["y_true"] = (
        subject_df["label"]
        .astype(int)
    )

    subject_df["y_pred"] = (
        subject_df["prob_PD"]
        >= threshold
    ).astype(int)

    return subject_df


def evaluate_subject_level(
    subject_df
):
    return calculate_metrics(
        subject_df[
            "y_true"
        ].values,
        subject_df[
            "y_pred"
        ].values,
        subject_df[
            "prob_PD"
        ].values,
    )


# =========================================================
# 14. Bootstrap CI
# =========================================================
def bootstrap_subject_metrics(
    subject_df,
    n_boot=2000,
    alpha=0.05,
    seed=40,
    stratified=True
):
    rng = np.random.default_rng(
        seed
    )

    metrics = {
        k: []
        for k in [
            "accuracy",
            "precision",
            "recall",
            "specificity",
            "f1",
            "auc"
        ]
    }

    control = (
        subject_df[
            subject_df.y_true == 0
        ]
        .reset_index(drop=True)
    )

    pd_group = (
        subject_df[
            subject_df.y_true == 1
        ]
        .reset_index(drop=True)
    )

    for _ in range(n_boot):
        if stratified:
            b0 = control.iloc[
                rng.integers(
                    0,
                    len(control),
                    len(control)
                )
            ]

            b1 = pd_group.iloc[
                rng.integers(
                    0,
                    len(pd_group),
                    len(pd_group)
                )
            ]

            boot = pd.concat(
                [b0, b1],
                ignore_index=True
            )

        else:
            boot = subject_df.iloc[
                rng.integers(
                    0,
                    len(subject_df),
                    len(subject_df)
                )
            ]

        m = calculate_metrics(
            boot.y_true,
            boot.y_pred,
            boot.prob_PD
        )

        for key in metrics:
            if not np.isnan(m[key]):
                metrics[key].append(
                    m[key]
                )

    out = {}

    lo_q = 100 * alpha / 2
    hi_q = 100 * (
        1 - alpha / 2
    )

    for key, vals in metrics.items():
        if vals:
            out[
                f"{key}_ci_low"
            ] = float(
                np.percentile(
                    vals,
                    lo_q
                )
            )

            out[
                f"{key}_ci_high"
            ] = float(
                np.percentile(
                    vals,
                    hi_q
                )
            )

        else:
            out[
                f"{key}_ci_low"
            ] = np.nan

            out[
                f"{key}_ci_high"
            ] = np.nan

    return out


# =========================================================
# 15. Training helpers
# =========================================================
def train_one_epoch(
    model,
    loader,
    criterion,
    optimizer,
    accumulation_steps=1
):
    model.train()

    total_loss = 0.0

    optimizer.zero_grad(
        set_to_none=True
    )

    for step, (
        images,
        labels
    ) in enumerate(
        loader,
        start=1
    ):
        images = images.to(
            DEVICE
        )

        labels = labels.to(
            DEVICE
        )

        outputs = model(images)

        loss = (
            criterion(
                outputs,
                labels
            )
            / accumulation_steps
        )

        loss.backward()

        if (
            step
            % accumulation_steps
            == 0
            or step == len(loader)
        ):
            optimizer.step()

            optimizer.zero_grad(
                set_to_none=True
            )

        total_loss += (
            loss.item()
            * accumulation_steps
        )

    return (
        total_loss
        / max(len(loader), 1)
    )


def make_criterion(
    train_df
):
    class_counts = (
        train_df["label"]
        .value_counts()
        .sort_index()
    )

    weights = np.asarray(
        [
            1.0 / max(
                class_counts.get(
                    0,
                    1
                ),
                1
            ),
            1.0 / max(
                class_counts.get(
                    1,
                    1
                ),
                1
            ),
        ],
        dtype=np.float32
    )

    weights = (
        weights
        / weights.sum()
        * 2
    )

    class_weights = torch.tensor(
        weights,
        dtype=torch.float32
    ).to(DEVICE)

    return nn.CrossEntropyLoss(
        weight=class_weights,
        label_smoothing=LABEL_SMOOTHING,
    )


def make_loader(
    df,
    image_size,
    shuffle
):
    return DataLoader(
        Gait16ChannelGAFDataset(df),
        batch_size=BATCH_SIZE_MAP[
            image_size
        ],
        shuffle=shuffle,
        num_workers=NUM_WORKERS,
    )


# =========================================================
# 16. One CV fold
# =========================================================
def train_cv_fold(
    model_name,
    image_size,
    fold,
    fold_dir,
    train_df,
    val_df
):
    set_seed(
        SEED + fold
    )

    train_loader = make_loader(
        train_df,
        image_size,
        True
    )

    val_loader = make_loader(
        val_df,
        image_size,
        False
    )

    criterion = make_criterion(
        train_df
    )

    model, init_status = build_model(
        model_name,
        image_size
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LR,
        weight_decay=WEIGHT_DECAY
    )

    accumulation_steps = (
        GRAD_ACCUM_MAP[
            image_size
        ]
    )

    best_auc = -1.0
    best_epoch = 0
    no_improve = 0

    safe_name = model_name.replace(
        "/",
        "_"
    )

    best_path = os.path.join(
        fold_dir,
        f"best_{safe_name}_{image_size}.pth"
    )

    logs = []

    for epoch in range(
        1,
        EPOCHS + 1
    ):
        train_loss = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            accumulation_steps
        )

        val_result = evaluate_recording_level(
            model,
            val_loader,
            criterion
        )

        logs.append({
            "fold":
                fold,
            "image_size":
                image_size,
            "model":
                model_name,
            "epoch":
                epoch,
            "train_loss":
                train_loss,
            "val_loss":
                val_result["loss"],
            "val_accuracy":
                val_result["accuracy"],
            "val_f1":
                val_result["f1"],
            "val_auc":
                val_result["auc"],
        })

        current_auc = (
            val_result["auc"]
            if not np.isnan(
                val_result["auc"]
            )
            else -1.0
        )

        print(
            f"CV Fold {fold} | "
            f"{image_size} | "
            f"{model_name} | "
            f"Epoch {epoch:03d} | "
            f"Val AUC {current_auc:.4f}"
        )

        if current_auc > best_auc:
            best_auc = current_auc
            best_epoch = epoch
            no_improve = 0

            torch.save(
                model.state_dict(),
                best_path
            )

        else:
            no_improve += 1

        if no_improve >= PATIENCE:
            break

    result = {
        "fold":
            fold,
        "image_size":
            image_size,
        "model":
            model_name,
        "train_subjects":
            train_df[
                "subject_id"
            ].nunique(),
        "validation_subjects":
            val_df[
                "subject_id"
            ].nunique(),
        "train_recordings":
            len(train_df),
        "validation_recordings":
            len(val_df),
        "best_epoch":
            best_epoch,
        "best_val_auc":
            best_auc,
        "pretraining_status":
            init_status,
    }

    del model
    gc.collect()

    if DEVICE == "cuda":
        torch.cuda.empty_cache()
    elif DEVICE == "mps":
        torch.mps.empty_cache()

    return result, logs


# =========================================================
# 17. Final training on ALL development subjects
# Uses CV-derived median best epoch.
# Independent test is untouched until this step.
# =========================================================
def train_final_and_test(
    model_name,
    image_size,
    output_dir,
    development_df,
    test_df,
    final_epochs
):
    print(
        "\n" + "=" * 90
    )
    print(
        f"FINAL MODEL | "
        f"{image_size} | "
        f"{model_name} | "
        f"epochs={final_epochs}"
    )
    print(
        "=" * 90
    )

    set_seed(SEED)

    train_loader = make_loader(
        development_df,
        image_size,
        True
    )

    test_loader = make_loader(
        test_df,
        image_size,
        False
    )

    criterion = make_criterion(
        development_df
    )

    model, init_status = build_model(
        model_name,
        image_size
    )

    total_params, trainable_params = (
        count_parameters(model)
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LR,
        weight_decay=WEIGHT_DECAY
    )

    accumulation_steps = (
        GRAD_ACCUM_MAP[
            image_size
        ]
    )

    start_time = time.time()

    for epoch in range(
        1,
        final_epochs + 1
    ):
        train_loss = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            accumulation_steps
        )

        print(
            f"FINAL | "
            f"{image_size} | "
            f"{model_name} | "
            f"Epoch {epoch:03d}/"
            f"{final_epochs} | "
            f"Loss {train_loss:.4f}"
        )

    training_time_sec = (
        time.time()
        - start_time
    )

    recording_result = (
        evaluate_recording_level(
            model,
            test_loader,
            criterion
        )
    )

    pred_df = (
        test_df.copy()
        .reset_index(drop=True)
    )

    pred_df["y_true"] = (
        recording_result[
            "y_true"
        ]
    )

    pred_df["y_pred"] = (
        recording_result[
            "y_pred"
        ]
    )

    pred_df["prob_PD"] = (
        recording_result[
            "y_prob"
        ]
    )

    safe_name = model_name.replace(
        "/",
        "_"
    )

    pred_path = os.path.join(
        output_dir,
        f"independent_test_predictions_recording_{safe_name}.csv"
    )

    pred_df.to_csv(
        pred_path,
        index=False
    )

    subject_df = aggregate_to_subject_level(
        pred_df,
        SUBJECT_THRESHOLD
    )

    subject_result = (
        evaluate_subject_level(
            subject_df
        )
    )

    subject_ci = bootstrap_subject_metrics(
        subject_df,
        n_boot=BOOTSTRAP_ITERATIONS,
        alpha=BOOTSTRAP_ALPHA,
        seed=SEED,
        stratified=BOOTSTRAP_STRATIFIED,
    )

    subject_pred_path = os.path.join(
        output_dir,
        f"independent_test_predictions_subject_{safe_name}.csv"
    )

    subject_df.to_csv(
        subject_pred_path,
        index=False
    )

    result = {
        "image_size":
            image_size,

        "model":
            model_name,

        "final_training_epochs":
            final_epochs,

        "development_subjects":
            development_df[
                "subject_id"
            ].nunique(),

        "development_recordings":
            len(development_df),

        "independent_test_subjects":
            test_df[
                "subject_id"
            ].nunique(),

        "independent_test_recordings":
            len(test_df),

        "pretraining_status":
            init_status,

        "recording_test_accuracy":
            recording_result[
                "accuracy"
            ],

        "recording_test_precision":
            recording_result[
                "precision"
            ],

        "recording_test_recall":
            recording_result[
                "recall"
            ],

        "recording_test_specificity":
            recording_result[
                "specificity"
            ],

        "recording_test_f1":
            recording_result[
                "f1"
            ],

        "recording_test_auc":
            recording_result[
                "auc"
            ],

        "subject_test_n":
            len(subject_df),

        "subject_test_accuracy":
            subject_result[
                "accuracy"
            ],

        "subject_test_precision":
            subject_result[
                "precision"
            ],

        "subject_test_recall":
            subject_result[
                "recall"
            ],

        "subject_test_specificity":
            subject_result[
                "specificity"
            ],

        "subject_test_f1":
            subject_result[
                "f1"
            ],

        "subject_test_auc":
            subject_result[
                "auc"
            ],

        "subject_confusion_matrix":
            json.dumps(
                subject_result[
                    "confusion_matrix"
                ].tolist()
            ),

        **subject_ci,

        "total_parameters":
            total_params,

        "trainable_parameters":
            trainable_params,

        "training_time_sec":
            training_time_sec,

        "test_used_in_cv":
            False,

        "test_used_for_model_selection":
            False,

        "selection_protocol":
            (
                "5-fold CV on development subjects only; "
                "final epoch count = median best epoch across folds"
            ),

        "recording_prediction_path":
            pred_path,

        "subject_prediction_path":
            subject_pred_path,

        "status":
            "completed",
    }

    del model
    gc.collect()

    if DEVICE == "cuda":
        torch.cuda.empty_cache()
    elif DEVICE == "mps":
        torch.mps.empty_cache()

    return result


# =========================================================
# 18. Pairwise McNemar on independent TEST subjects
# =========================================================
def exact_mcnemar(
    y_true,
    pred_a,
    pred_b
):
    y_true = np.asarray(
        y_true
    )

    a = np.asarray(
        pred_a
    )

    b = np.asarray(
        pred_b
    )

    a_correct = (
        a == y_true
    )

    b_correct = (
        b == y_true
    )

    n01 = int(
        np.sum(
            a_correct
            & ~b_correct
        )
    )

    n10 = int(
        np.sum(
            ~a_correct
            & b_correct
        )
    )

    n = n01 + n10

    if n == 0:
        return (
            n01,
            n10,
            1.0
        )

    p = binomtest(
        min(n01, n10),
        n=n,
        p=0.5,
        alternative="two-sided"
    ).pvalue

    return (
        n01,
        n10,
        float(p)
    )


def run_pairwise_subject_mcnemar(
    final_results
):
    rows = []

    for image_size in IMAGE_SIZES:
        completed = [
            r
            for r in final_results
            if r.get(
                "image_size"
            ) == image_size
            and r.get(
                "status"
            ) == "completed"
        ]

        by_model = {
            r["model"]:
                r
            for r in completed
        }

        for (
            model_a,
            model_b
        ) in itertools.combinations(
            sorted(by_model),
            2
        ):
            a = pd.read_csv(
                by_model[
                    model_a
                ][
                    "subject_prediction_path"
                ]
            )

            b = pd.read_csv(
                by_model[
                    model_b
                ][
                    "subject_prediction_path"
                ]
            )

            merged = (
                a[
                    [
                        "subject_id",
                        "y_true",
                        "y_pred"
                    ]
                ]
                .merge(
                    b[
                        [
                            "subject_id",
                            "y_pred"
                        ]
                    ],
                    on="subject_id",
                    suffixes=(
                        "_a",
                        "_b"
                    )
                )
            )

            n01, n10, p = exact_mcnemar(
                merged[
                    "y_true"
                ],
                merged[
                    "y_pred_a"
                ],
                merged[
                    "y_pred_b"
                ]
            )

            rows.append({
                "image_size":
                    image_size,

                "model_a":
                    model_a,

                "model_b":
                    model_b,

                "n_model_a_correct_model_b_wrong":
                    n01,

                "n_model_a_wrong_model_b_correct":
                    n10,

                "mcnemar_exact_p":
                    p,
            })

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(
        rows
    )

    reject, p_holm, _, _ = (
        multipletests(
            df[
                "mcnemar_exact_p"
            ].values,
            alpha=0.05,
            method="holm"
        )
    )

    df[
        "holm_adjusted_p"
    ] = p_holm

    df[
        "significant_after_holm_0.05"
    ] = reject

    return df


# =========================================================
# 19. Main
# =========================================================
def main():
    set_seed(SEED)

    os.makedirs(
        OUTPUT_ROOT,
        exist_ok=True
    )

    save_environment_info()

    all_cv_results = []
    all_cv_logs = []
    all_final_results = []
    all_test_split_summaries = []
    all_cv_partition_summaries = []

    for image_size in IMAGE_SIZES:
        print(
            "\n" + "#" * 100
        )
        print(
            f"IMAGE SIZE: "
            f"{image_size}x{image_size}"
        )
        print(
            "#" * 100
        )

        meta, output_dir = (
            create_array_dataset(
                image_size
            )
        )

        (
            development_subjects,
            test_subjects,
            development_df,
            test_df,
            test_split_summary
        ) = make_independent_test_split(
            meta,
            output_dir
        )

        test_split_summary[
            "image_size"
        ] = image_size

        all_test_split_summaries.extend(
            test_split_summary.to_dict(
                "records"
            )
        )

        fold_data, fold_summary = (
            make_development_folds(
                development_subjects,
                development_df,
                output_dir
            )
        )

        fold_summary[
            "image_size"
        ] = image_size

        all_cv_partition_summaries.extend(
            fold_summary.to_dict(
                "records"
            )
        )

        for model_name in MODEL_LIST:
            model_fold_results = []

            for fold_info in fold_data:
                try:
                    fold_result, fold_logs = train_cv_fold(
                        model_name=model_name,
                        image_size=image_size,
                        fold=fold_info[
                            "fold"
                        ],
                        fold_dir=fold_info[
                            "fold_dir"
                        ],
                        train_df=fold_info[
                            "train_df"
                        ],
                        val_df=fold_info[
                            "val_df"
                        ],
                    )

                    all_cv_results.append(
                        fold_result
                    )

                    model_fold_results.append(
                        fold_result
                    )

                    all_cv_logs.extend(
                        fold_logs
                    )

                except Exception as e:
                    print(
                        f"CV FAILED: "
                        f"size={image_size}, "
                        f"model={model_name}, "
                        f"fold={fold_info['fold']}: "
                        f"{e}"
                    )

            if not model_fold_results:
                all_final_results.append({
                    "image_size":
                        image_size,
                    "model":
                        model_name,
                    "status":
                        "failed",
                    "error":
                        "All CV folds failed"
                })
                continue

            best_epochs = [
                int(
                    r[
                        "best_epoch"
                    ]
                )
                for r in model_fold_results
                if r[
                    "best_epoch"
                ] > 0
            ]

            if not best_epochs:
                all_final_results.append({
                    "image_size":
                        image_size,
                    "model":
                        model_name,
                    "status":
                        "failed",
                    "error":
                        "No valid best epochs from CV"
                })
                continue

            final_epochs = max(
                1,
                int(
                    round(
                        np.median(
                            best_epochs
                        )
                    )
                )
            )

            try:
                final_result = (
                    train_final_and_test(
                        model_name=model_name,
                        image_size=image_size,
                        output_dir=output_dir,
                        development_df=development_df,
                        test_df=test_df,
                        final_epochs=final_epochs
                    )
                )

                final_result[
                    "cv_best_epoch_mean"
                ] = float(
                    np.mean(
                        best_epochs
                    )
                )

                final_result[
                    "cv_best_epoch_sd"
                ] = float(
                    np.std(
                        best_epochs,
                        ddof=1
                    )
                    if len(
                        best_epochs
                    ) > 1
                    else 0.0
                )

                final_result[
                    "cv_best_val_auc_mean"
                ] = float(
                    np.mean(
                        [
                            r[
                                "best_val_auc"
                            ]
                            for r in model_fold_results
                        ]
                    )
                )

                final_result[
                    "cv_best_val_auc_sd"
                ] = float(
                    np.std(
                        [
                            r[
                                "best_val_auc"
                            ]
                            for r in model_fold_results
                        ],
                        ddof=1
                    )
                    if len(
                        model_fold_results
                    ) > 1
                    else 0.0
                )

                all_final_results.append(
                    final_result
                )

            except Exception as e:
                print(
                    f"FINAL TEST FAILED: "
                    f"size={image_size}, "
                    f"model={model_name}: "
                    f"{e}"
                )

                all_final_results.append({
                    "image_size":
                        image_size,
                    "model":
                        model_name,
                    "status":
                        "failed",
                    "error":
                        str(e)
                })

            # Save progress continuously
            pd.DataFrame(
                all_cv_results
            ).to_csv(
                os.path.join(
                    OUTPUT_ROOT,
                    "5fold_development_results_all_folds.csv"
                ),
                index=False
            )

            pd.DataFrame(
                all_cv_logs
            ).to_csv(
                os.path.join(
                    OUTPUT_ROOT,
                    "5fold_development_training_logs.csv"
                ),
                index=False
            )

            pd.DataFrame(
                all_final_results
            ).to_csv(
                os.path.join(
                    OUTPUT_ROOT,
                    "independent_test_final_results.csv"
                ),
                index=False
            )

    # Summarize CV
    cv_df = pd.DataFrame(
        all_cv_results
    )

    if not cv_df.empty:
        cv_summary = (
            cv_df.groupby(
                [
                    "image_size",
                    "model"
                ],
                as_index=False
            )
            .agg(
                folds_completed=(
                    "fold",
                    "count"
                ),
                best_epoch_mean=(
                    "best_epoch",
                    "mean"
                ),
                best_epoch_sd=(
                    "best_epoch",
                    "std"
                ),
                val_auc_mean=(
                    "best_val_auc",
                    "mean"
                ),
                val_auc_sd=(
                    "best_val_auc",
                    "std"
                ),
            )
        )

        cv_summary.to_csv(
            os.path.join(
                OUTPUT_ROOT,
                "5fold_development_results_mean_sd.csv"
            ),
            index=False
        )

    pd.DataFrame(
        all_test_split_summaries
    ).to_csv(
        os.path.join(
            OUTPUT_ROOT,
            "independent_test_split_summary_all_sizes.csv"
        ),
        index=False
    )

    pd.DataFrame(
        all_cv_partition_summaries
    ).to_csv(
        os.path.join(
            OUTPUT_ROOT,
            "5fold_development_partition_summary_all_sizes.csv"
        ),
        index=False
    )

    final_df = pd.DataFrame(
        all_final_results
    )

    final_df.to_csv(
        os.path.join(
            OUTPUT_ROOT,
            "independent_test_final_results.csv"
        ),
        index=False
    )

    mcnemar_df = (
        run_pairwise_subject_mcnemar(
            all_final_results
        )
    )

    mcnemar_df.to_csv(
        os.path.join(
            OUTPUT_ROOT,
            "independent_test_subject_level_mcnemar_holm.csv"
        ),
        index=False
    )

    print(
        "\nSaved outputs:"
    )

    print(
        "- environment_and_protocol.json"
    )
    print(
        "- independent_test_split_summary_all_sizes.csv"
    )
    print(
        "- 5fold_development_partition_summary_all_sizes.csv"
    )
    print(
        "- 5fold_development_results_all_folds.csv"
    )
    print(
        "- 5fold_development_results_mean_sd.csv"
    )
    print(
        "- independent_test_final_results.csv"
    )
    print(
        "- independent_test_subject_level_mcnemar_holm.csv"
    )


if __name__ == "__main__":
    main()
