# Supplementary Material

## Exploratory resting-state EEG associations with anxiety and perceived social support in two distinct samples

Version 1.0.0; assembled 2026-07-28.

## S1. Scope

This archive contains aggregate, nonidentifying results, sanitized analysis code, preprocessing/QC manifests, and complete permutation metadata. The participant was the inferential unit.

## S2. Why 51 selections produced 50 targets

The 51 rows are selected local interaction-model rows. Target-level transfer used the key `representation + local_outcome + eeg_id`. `TARGET__DELTA_OC_MINUS_OA__MSPSS__EEG23` (alpha C4, EC−EO) occurred in both W3 and W4 models. It therefore contributes two selection rows but one unique feature target. It remains present in both corresponding frozen signatures.

## S3. Frozen specification

Frozen elements were component identities and directional equal-magnitude weights. Standardization parameters were not transferred. Each component was standardized using the mean and sample standard deviation of the analytical sample in which the signature was scored, then the frozen weights were applied.

## S4. Permutations

All empirical values used `(b + 1)/(B + 1)` with `B = 10,000`. Local correlation and selection-aware analyses used seed `20260726`. External LEMON analyses used seed `20260727`. The full manifest is `tables/S01_permutation_manifest.csv` and `manifests/permutation_manifest.json`.

## S5. Machine-readable contents

- `Supplementary_Tables.xlsx`: formatted workbook mirroring the audit tables.
- `tables/`: complete aggregate result tables S01–S16.
- `manifests/`: permutation, runtime, preprocessing/QC, G*Power, and checksum manifests.
- `code/`: sanitized Python analysis scripts.
- `environment/requirements.txt`: pinned statistical environment.
