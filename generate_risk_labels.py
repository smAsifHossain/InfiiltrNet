"""Distance-based surrogate-label generation for InfiltrNet (BraTS 2020/2025).

Portable extraction of the label-generation rules in the original BraTS 2025
experimental pipeline. BraTS 2020 uses the corresponding enhancing-tumor code.
This module provides label construction only; it is not biological ground truth.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from scipy.ndimage import distance_transform_edt

HIGH_RISK_DISTANCE_MM = 10.0
MEDIUM_RISK_DISTANCE_MM = 20.0
VOXEL_SIZE_MM = 1.0  # Experimental BraTS volumes: 1-mm isotropic.
ENHANCING_LABEL = {"2020": 4, "2025": 3}


def make_risk_labels(seg: np.ndarray, flair: np.ndarray, brats_version: str) -> np.ndarray:
    """Return the experimental four-class distance-based surrogate map.

    Labels: 0 = tumor core / non-brain; 1 = >20 mm; 2 = (10,20] mm;
    3 = edema OR (0,10] mm beyond the complete-tumor boundary.
    Distances are computed outward from core union edema at 1-mm spacing.
    """
    version = str(brats_version)
    if version not in ENHANCING_LABEL:
        raise ValueError("brats_version must be '2020' or '2025'")
    if seg.ndim != 3 or flair.shape != seg.shape:
        raise ValueError("seg and flair must be matching 3-D arrays")

    necrotic = seg == 1
    edema = seg == 2
    enhancing = seg == ENHANCING_LABEL[version]
    tumor_core = necrotic | enhancing
    tumor_all = tumor_core | edema
    brain_mask = flair > 0

    if not np.any(tumor_all):
        return np.zeros(seg.shape, dtype=np.float32)

    dist = distance_transform_edt(~tumor_all) * VOXEL_SIZE_MM
    infil = np.zeros(seg.shape, dtype=np.int8)

    # Preserve the original zone-assignment order and inclusion boundaries.
    infil[edema | ((dist > 0) & (dist <= HIGH_RISK_DISTANCE_MM)
                   & ~tumor_core & brain_mask)] = 3
    infil[(dist > HIGH_RISK_DISTANCE_MM) & (dist <= MEDIUM_RISK_DISTANCE_MM)
          & ~tumor_core & brain_mask] = 2
    infil[(dist > MEDIUM_RISK_DISTANCE_MM) & ~tumor_core & brain_mask] = 1

    return infil.astype(np.float32)


def generate_label_file(seg_path: Path, flair_path: Path, output_path: Path,
                        brats_version: str) -> None:
    """Read aligned NIfTI inputs and save labels using the segmentation affine."""
    import nibabel as nib

    seg_nii = nib.load(str(seg_path))
    seg = seg_nii.get_fdata().astype(int)
    flair = nib.load(str(flair_path)).get_fdata()
    labels = make_risk_labels(seg, flair, brats_version)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(nib.Nifti1Image(labels, seg_nii.affine), str(output_path))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seg", type=Path, required=True, help="BraTS segmentation NIfTI")
    parser.add_argument("--flair", type=Path, required=True, help="Aligned T2-FLAIR NIfTI")
    parser.add_argument("--output", type=Path, required=True, help="Output risk-label NIfTI")
    parser.add_argument("--brats-version", required=True, choices=sorted(ENHANCING_LABEL))
    args = parser.parse_args()
    generate_label_file(args.seg, args.flair, args.output, args.brats_version)


if __name__ == "__main__":
    main()
