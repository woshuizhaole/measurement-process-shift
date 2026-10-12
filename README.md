# Measurement-process shift in cross-hospital ICU early-warning models

Code for the study *Auditing the measurement process before cross-hospital deployment of ICU
early-warning systems: an external-validation and transferability study across two ICU systems and
147 hospitals.*

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

Scripts read a project root `P` (default: the repository root) and expect the raw data under
`./data/raw/`, the prepared arrays under `./data/processed/`, and the external cohorts under
`./data/external/`. Run scripts from the repository root.

## Data

The study uses only de-identified, openly available data; none is redistributed here.

- PhysioNet/Computing in Cardiology Challenge 2019, training sets A and B — the two hospital
  systems (A: Beth Israel Deaconess Medical Center; B: Emory University Hospital), with the
  Challenge sepsis label and Utility — https://physionet.org/content/challenge-2019/
- PhysioNet/Computing in Cardiology Challenge 2012 — single-hospital negative control —
  https://physionet.org/content/challenge-2012/
- WiDS Datathon 2020 — 147-hospital universality cohort — https://physionet.org/content/widsdatathon2020/

Place the Challenge 2019 training sets under `./data/raw/cinc2019/training/` (`training_setA/`,
`training_setB/`), the extracted Challenge 2012 records under `./data/interim/cinc2012/`, and the
WiDS file under `./data/external/wids2020/`, then run the scripts in `code/pipeline/` to produce the
processed arrays under `./data/processed/`. The descriptive script `code/analysis/73_eicu_labs.py`
reads the open eICU-CRD demo; it is not used for any reported result.

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
