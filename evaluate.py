"""Evaluate InfiltrNet with TTA and explicit pre/post-processing metrics."""

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import torch

from data import create_dataloaders
from infiltrnet import InfiltrNet
from metrics import build_legacy_metrics, per_patient_zone_metrics
from profiles import get_profile
from utils import (
    class_map_to_scores,
    get_device,
    postprocess,
    predict_with_tta,
    save_json,
    set_seed,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, choices=["brats2020", "brats2025"])
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--labels-dir", required=True)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--min-component-size", type=int, default=500)
    args = parser.parse_args()

    profile = get_profile(args.dataset)
    set_seed(42)
    device = get_device()

    run_dir = Path(args.run_dir)
    checkpoint = run_dir / "checkpoints" / "best.pth"
    if not checkpoint.exists():
        raise FileNotFoundError(checkpoint)

    model = InfiltrNet(in_channels=4, out_channels=4, sf=24, cf=32).to(device)
    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    _, _, test_loader, split = create_dataloaders(
        dataset=args.dataset,
        data_root=args.data_root,
        labels_dir=args.labels_dir,
        split_file=run_dir / "data_split.json",
        num_workers=args.num_workers,
    )

    legacy_raw = build_legacy_metrics(args.dataset)
    postprocessed_metrics = build_legacy_metrics(args.dataset)
    patient_rows_raw = []
    patient_rows_pp = []

    for idx, batch in enumerate(test_loader):
        images = batch["image"]
        labels = batch["label"]
        pid = batch.get("patient_id", [f"patient_{idx}"])
        if isinstance(pid, (list, tuple)):
            pid = pid[0]

        avg_probs = predict_with_tta(
            model,
            images,
            device,
            roi_size=profile.patch_size,
            sw_batch_size=profile.sw_batch_size,
        )

        pred_raw = torch.argmax(avg_probs, dim=1)[0].numpy().astype(np.int16)
        gt_map = labels[0, 0].numpy().astype(np.int16)
        pred_pp = postprocess(pred_raw, min_size=args.min_component_size)

        # Reproduces the original segmentation-metric path:
        # TTA prediction before connected-component/hole-filling post-processing.
        legacy_raw.update(avg_probs, labels.float())

        # Explicit post-processing metric path for the revised analysis.
        pp_scores = class_map_to_scores(pred_pp, num_classes=4)
        postprocessed_metrics.update(pp_scores, labels.float())

        raw_row = {"patient_id": str(pid), **per_patient_zone_metrics(pred_raw, gt_map)}
        pp_row = {"patient_id": str(pid), **per_patient_zone_metrics(pred_pp, gt_map)}
        patient_rows_raw.append(raw_row)
        patient_rows_pp.append(pp_row)

        print(f"[{idx + 1}/{len(test_loader)}] {pid}")

    result = {
        "dataset": args.dataset,
        "checkpoint_epoch": int(ckpt.get("epoch", -1)),
        "checkpoint_metric": float(ckpt.get("metric", float("nan"))),
        "legacy_pre_postprocessing": legacy_raw.compute(),
        "postprocessed": postprocessed_metrics.compute(),
        "postprocessing": {
            "min_component_size_voxels": args.min_component_size,
            "hole_filling": True,
        },
        "test_split_size": len(split["test"]),
    }
    save_json(result, run_dir / "evaluation.json")

    pd.DataFrame(patient_rows_raw).to_csv(
        run_dir / "per_patient_metrics_raw.csv",
        index=False,
        float_format="%.6f",
    )
    pd.DataFrame(patient_rows_pp).to_csv(
        run_dir / "per_patient_metrics_postprocessed.csv",
        index=False,
        float_format="%.6f",
    )
    print(result)


if __name__ == "__main__":
    main()
