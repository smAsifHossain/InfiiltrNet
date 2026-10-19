"""Dataset loaders for the BraTS 2020 and BraTS 2025 experiments."""

import json
import random
from pathlib import Path
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from monai.data import CacheDataset, DataLoader
from monai.transforms import (
    Compose,
    LoadImaged,
    EnsureChannelFirstd,
    Orientationd,
    NormalizeIntensityd,
    ConcatItemsd,
    CropForegroundd,
    SpatialPadd,
    RandCropByPosNegLabeld,
    RandFlipd,
    RandRotate90d,
    RandScaleIntensityd,
    RandShiftIntensityd,
    EnsureTyped,
)

from profiles import get_profile


def _load_or_make_split_2020(patient_ids, split_file, seed=42):
    split_file = Path(split_file)
    if split_file.exists():
        return json.loads(split_file.read_text())

    train_ids, test_ids = train_test_split(
        patient_ids,
        test_size=0.1,
        random_state=seed,
    )
    train_ids, val_ids = train_test_split(
        train_ids,
        test_size=0.1 / (0.8 + 0.1),
        random_state=seed,
    )
    split = {"train": train_ids, "val": val_ids, "test": test_ids}
    split_file.parent.mkdir(parents=True, exist_ok=True)
    split_file.write_text(json.dumps(split, indent=2))
    return split


def _load_or_make_split_2025(patient_ids, split_file, seed=42):
    split_file = Path(split_file)
    if split_file.exists():
        return json.loads(split_file.read_text())

    patient_ids = list(sorted(patient_ids))
    random.seed(seed)
    random.shuffle(patient_ids)

    n = len(patient_ids)
    n_train = int(n * 0.80)
    n_val = int(n * 0.10)

    split = {
        "train": patient_ids[:n_train],
        "val": patient_ids[n_train:n_train + n_val],
        "test": patient_ids[n_train + n_val:],
    }
    split_file.parent.mkdir(parents=True, exist_ok=True)
    split_file.write_text(json.dumps(split, indent=2))
    return split


def _build_2020_list(data_root, labels_dir, patient_ids):
    data_root = Path(data_root)
    labels_dir = Path(labels_dir)
    rows = []
    for pid in patient_ids:
        p = data_root / pid
        row = {
            "flair": str(p / f"{pid}_flair.nii"),
            "t1": str(p / f"{pid}_t1.nii"),
            "t1ce": str(p / f"{pid}_t1ce.nii"),
            "t2": str(p / f"{pid}_t2.nii"),
            "label": str(labels_dir / f"{pid}_infiltration.nii.gz"),
            "patient_id": pid,
        }
        if all(Path(row[k]).exists() for k in ("flair", "t1", "t1ce", "t2", "label")):
            rows.append(row)
    return rows


def _build_2025_list(data_root, labels_dir, patient_ids):
    data_root = Path(data_root)
    labels_dir = Path(labels_dir)
    rows = []
    for pid in patient_ids:
        p = data_root / pid
        row = {
            "t1c": str(p / f"{pid}-t1c.nii.gz"),
            "t1n": str(p / f"{pid}-t1n.nii.gz"),
            "t2f": str(p / f"{pid}-t2f.nii.gz"),
            "t2w": str(p / f"{pid}-t2w.nii.gz"),
            "label": str(labels_dir / f"{pid}_infiltration.nii.gz"),
            "patient_id": pid,
        }
        if all(Path(row[k]).exists() for k in ("t1c", "t1n", "t2f", "t2w", "label")):
            rows.append(row)
    return rows


def create_dataloaders(
    dataset,
    data_root,
    labels_dir,
    split_file,
    cache_rate=None,
    num_workers=2,
):
    profile = get_profile(dataset)
    data_root = Path(data_root)
    labels_dir = Path(labels_dir)

    if dataset == "brats2020":
        patient_ids = sorted(
            p.name
            for p in data_root.glob("BraTS20_Training_*")
            if (labels_dir / f"{p.name}_infiltration.nii.gz").exists()
        )
        split = _load_or_make_split_2020(patient_ids, split_file, seed=42)
        build = _build_2020_list

        mod_keys = ["flair", "t1", "t1ce", "t2"]
        all_keys = mod_keys + ["label"]

        train_transforms = Compose([
            LoadImaged(keys=all_keys, image_only=False),
            EnsureChannelFirstd(keys=all_keys),
            Orientationd(keys=all_keys, axcodes="RAS"),
            NormalizeIntensityd(keys=mod_keys, nonzero=True, channel_wise=True),
            ConcatItemsd(keys=mod_keys, name="image", dim=0),
            CropForegroundd(keys=["image", "label"], source_key="image", margin=10),
            SpatialPadd(keys=["image", "label"], spatial_size=profile.patch_size, mode="constant"),
            RandCropByPosNegLabeld(
                keys=["image", "label"],
                label_key="label",
                spatial_size=profile.patch_size,
                pos=0.6,
                neg=0.4,
                num_samples=profile.num_samples,
            ),
            RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
            RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=1),
            RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=2),
            RandRotate90d(keys=["image", "label"], prob=0.3, max_k=3),
            RandScaleIntensityd(keys=["image"], factors=0.1, prob=0.3),
            RandShiftIntensityd(keys=["image"], offsets=0.1, prob=0.3),
            EnsureTyped(keys=["image", "label"], dtype=torch.float32),
        ])
        val_transforms = Compose([
            LoadImaged(keys=all_keys, image_only=False),
            EnsureChannelFirstd(keys=all_keys),
            Orientationd(keys=all_keys, axcodes="RAS"),
            NormalizeIntensityd(keys=mod_keys, nonzero=True, channel_wise=True),
            ConcatItemsd(keys=mod_keys, name="image", dim=0),
            CropForegroundd(keys=["image", "label"], source_key="image", margin=10),
            SpatialPadd(keys=["image", "label"], spatial_size=profile.patch_size, mode="constant"),
            EnsureTyped(keys=["image", "label"], dtype=torch.float32),
        ])
        train_cache, eval_cache = (0.1, 1.0) if cache_rate is None else (cache_rate, cache_rate)

    elif dataset == "brats2025":
        patient_ids = sorted(
            p.name
            for p in data_root.iterdir()
            if p.is_dir() and (labels_dir / f"{p.name}_infiltration.nii.gz").exists()
        )
        split = _load_or_make_split_2025(patient_ids, split_file, seed=42)
        build = _build_2025_list

        mod_keys = ["t1c", "t1n", "t2f", "t2w"]
        all_keys = mod_keys + ["label"]
        train_transforms = Compose([
            LoadImaged(keys=all_keys, image_only=True),
            EnsureChannelFirstd(keys=all_keys),
            Orientationd(keys=all_keys, axcodes="RAS"),
            NormalizeIntensityd(keys=mod_keys, nonzero=True, channel_wise=True),
            ConcatItemsd(keys=mod_keys, name="image"),
            RandCropByPosNegLabeld(
                keys=["image", "label"],
                label_key="label",
                spatial_size=profile.patch_size,
                pos=1,
                neg=1,
                num_samples=profile.num_samples,
            ),
            RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
            RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=1),
            RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=2),
            RandRotate90d(keys=["image", "label"], prob=0.5, max_k=3),
            EnsureTyped(keys=["image", "label"]),
        ])
        val_transforms = Compose([
            LoadImaged(keys=all_keys, image_only=True),
            EnsureChannelFirstd(keys=all_keys),
            Orientationd(keys=all_keys, axcodes="RAS"),
            NormalizeIntensityd(keys=mod_keys, nonzero=True, channel_wise=True),
            ConcatItemsd(keys=mod_keys, name="image"),
            EnsureTyped(keys=["image", "label"]),
        ])
        train_cache = eval_cache = 0.1 if cache_rate is None else cache_rate
    else:
        raise ValueError("dataset must be 'brats2020' or 'brats2025'")

    train_list = build(data_root, labels_dir, split["train"])
    val_list = build(data_root, labels_dir, split["val"])
    test_list = build(data_root, labels_dir, split["test"])

    train_ds = CacheDataset(train_list, transform=train_transforms, cache_rate=train_cache, num_workers=num_workers)
    val_ds = CacheDataset(val_list, transform=val_transforms, cache_rate=eval_cache, num_workers=num_workers)
    test_ds = CacheDataset(test_list, transform=val_transforms, cache_rate=eval_cache, num_workers=num_workers)

    train_loader = DataLoader(
        train_ds,
        batch_size=profile.batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=1,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=1,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    return train_loader, val_loader, test_loader, split
