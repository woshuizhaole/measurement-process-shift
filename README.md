# Measurement-process shift in cross-hospital ICU early-warning models

Code for the study *A measurement-process fingerprint drives the cross-hospital collapse of ICU
early-warning models.*

A hospital's measurement process — which variables are ordered, and how often — is a
machine-readable institutional fingerprint. This repository contains the analysis pipeline that
(1) quantifies how well the measurement mask identifies the source hospital, (2) measures
cross-hospital transfer with the official deployment Utility rather than discrimination alone,
(3) decomposes the source-to-target gap into measurement, value, and prevalence channels, and
(4) evaluates repair methods, label-free recalibration, and a variable-level equivalence test.

## Repository layout

```
code/
  utils/        core methods: models, dual-channel (value + measurement mask) featurisation,
                the official Utility scorer, repair methods, and the paired bootstrap.
  analysis/     the numbered analysis stages that produce the reported results
                (fingerprint identification, transfer, channel decomposition, repair,
                 equivalence testing, controls, and the multi-hospital universality analysis).
  pipeline/     data preprocessing, cohort splits, and normalisation.
environment.yml, pip_freeze.txt   the analysis environment.
```

Scripts read a project root `P` (default: the repository root) and expect the prepared data under
`./data/processed/` and the external cohorts under `./data/external/`. Run scripts from the
repository root.

## Data

The study uses only de-identified, publicly available data, obtained from PhysioNet under the
respective data-use agreements; none is redistributed here.

- MIMIC-IV — https://physionet.org/content/mimiciv/
- eICU Collaborative Research Database — https://physionet.org/content/eicu-crd/
- PhysioNet/CinC Challenge 2019 (early-warning label and Utility) — https://physionet.org/content/challenge-2019/
- WiDS Datathon 2020 (147-hospital universality cohort) — https://physionet.org/content/widsdatathon2020/

Place the raw downloads under `./data/external/` and run the scripts in `code/pipeline/` to produce
the processed arrays under `./data/processed/`.

## Environment

```
conda env create -f environment.yml
conda activate ops-shift
```

Core dependencies: numpy, scipy, scikit-learn, lightgbm, pytorch, statsmodels, polars, matplotlib.

## Reproducing the main results

Run from the repository root, in stage order. Representative entry points:

| Result | Script |
|---|---|
| Fingerprint identifiability (mask → hospital) | `code/analysis/10_probe_ladder.py`, `code/analysis/fig2_compute.py` |
| 147-hospital universality | `code/analysis/71b_wids_147.py` |
| Cross-hospital transfer and deployment Utility | `code/analysis/40_ops_repair_main.py`, `code/analysis/74_sixdomain_lodo_utility.py` |
| Ranking-vs-threshold and channel decomposition | `code/analysis/23_channel_decomp.py`, `code/analysis/23b_channel_decomp_utility.py` |
| Variable-level equivalence (TOST) | `code/analysis/30b_gamma_T2_tost.py` |
| Repair, backbones, and controls | `code/analysis/40c_ops_lgb_rigorous.py`, `code/analysis/40d_ops_gru_rigorous.py`, `code/analysis/70_cinc2012_negcontrol.py`, `code/analysis/60_ucar_positive_control.py` |

Domain-generalisation baselines (`code/analysis/50_dg_baselines.py`) follow standard
leave-one-domain-out training; the recurrent backbone uses a GRU with per-variable decay.

## License

Released under the MIT License (see `LICENSE`).
