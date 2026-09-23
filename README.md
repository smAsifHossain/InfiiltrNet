# InfiltrNet — distance-based surrogate label generation

This repository contains the label-generation implementation used to define the supervised **distance-based surrogate risk maps** in the InfiltrNet manuscript (MDPI *Signals*, manuscript ID `signals-4504052`). These maps are constructed geometrically from BraTS annotations. **They are not measurements or calibrated probabilities of microscopic tumor-cell infiltration.**

The implementation is a portable extraction of the label-assignment logic in the author's provided BraTS 2025 experimental pipeline, with support for the corresponding BraTS 2020 segmentation-label convention. The historical BraTS 2020 training script has not been independently audited against this portable version. The repository currently provides label construction only, not model training weights or biological validation.

## Input conventions

| Dataset | Necrotic/core | Edema | Enhancing | FLAIR filename convention |
| --- | --- | --- | --- | --- |
| BraTS 2020 | 1 | 2 | 4 | `*_flair.nii` (or corresponding NIfTI file) |
| BraTS 2025 | 1 | 2 | 3 | `*-t2f.nii.gz` |

Inputs must be spatially aligned. The experimental volumes use 1-mm isotropic spacing. The script requires an explicit FLAIR image to derive the brain mask from positive-intensity voxels. In the original BraTS 2025 pipeline, a missing FLAIR file triggered an all-ones brain-mask fallback; this standalone interface deliberately requires FLAIR so missing input cannot silently alter the output.

## Definition

Let `tumor_core = (necrotic OR enhancing)`, `complete_tumor = (tumor_core OR edema)` and `brain_mask = (FLAIR > 0)`. Euclidean distance is computed *outside* `complete_tumor` using `distance_transform_edt(~complete_tumor) * 1.0` (millimetres). The classes are assigned in the same order and with the same inequalities as the original code.

| Output | Definition |
| --- | --- |
| 3 — high surrogate zone | Edema OR (0, 10] mm outside the complete-tumor boundary, within brain and outside tumor core |
| 2 — medium surrogate zone | (10, 20] mm outside that boundary, within brain |
| 1 — low surrogate zone | >20 mm outside that boundary, within brain |
| 0 | Tumor core or non-brain/background |

## Usage

~~~bash
python -m pip install -r requirements.txt
python generate_risk_labels.py \
  --seg /path/to/patient-seg.nii.gz \
  --flair /path/to/patient-t2f.nii.gz \
  --output /path/to/patient_infiltration.nii.gz \
  --brats-version 2025
~~~

For BraTS 2020, use `--brats-version 2020`, its corresponding segmentation, and its FLAIR image. The saved label map retains the segmentation NIfTI's affine transform and uses the original pipeline's float32 output convention.

## Verification

~~~bash
python -m unittest discover -s tests -v
~~~

The tests compare the BraTS 2025 output against a separate transcription of the supplied experimental rule on synthetic inputs, check exact 10-mm and 20-mm boundaries, the edema/tumor-core handling, and the 2020 enhancing-label conversion. Tests are synthetic implementation checks, not a patient-level validation of the published results.

## Source and citation

The algorithm is derived from the author's provided `data_pipeline_brats2025.py` and `config_brats2025.py`. Please cite the InfiltrNet manuscript when referring to this surrogate label definition.
