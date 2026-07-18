#!/usr/bin/env python
"""Stage 0 :: 生成 data/manifests/DATA_MANIFEST.tsv"""
import os, hashlib, glob, datetime, subprocess

P = "."
MAN = f"{P}/data/manifests/DATA_MANIFEST.tsv"
os.makedirs(os.path.dirname(MAN), exist_ok=True)

COLS = ["dataset_name", "source_url", "download_method", "downloaded_at", "raw_path",
        "file_size", "sha256_or_note", "expected_files", "actual_files", "status", "notes"]


def sha256_head(path, limit=None):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(1 << 20)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def du(path):
    try:
        return subprocess.check_output(["du", "-sh", path]).split()[0].decode()
    except Exception:
        return "?"


def count(path, pat="*"):
    return len(glob.glob(os.path.join(path, pat)))


now = datetime.datetime.now().isoformat(timespec="seconds")
rows = []

# CinC2019
a = f"{P}/data/raw/cinc2019/training/training_setA"
b = f"{P}/data/raw/cinc2019/training/training_setB"
rows.append(["CinC2019", "s3://physionet-open/challenge-2019/1.0.0/", "aws s3 sync --no-sign-request",
             now, f"{P}/data/raw/cinc2019", du(f"{P}/data/raw/cinc2019"),
             "SHA256SUMS.txt verified (LICENSE/svg OK)", "20336+20000 psv",
             f"{count(a,'*.psv')}+{count(b,'*.psv')}",
             "OK" if count(a, '*.psv') == 20336 and count(b, '*.psv') == 20000 else "CHECK",
             "setA=BIDMC, setB=Emory; license ODbL(shipped)/page CC BY 4.0"])

# CinC2012
c12 = f"{P}/data/raw/cinc2012"
inter = f"{P}/data/interim/cinc2012"
na = count(f"{inter}/set-a", "*.txt")
nb = count(f"{inter}/set-b", "*.txt")
nc = count(f"{inter}/set-c", "*.txt")
rows.append(["CinC2012", "s3://physionet-open/challenge-2012/1.0.0/", "aws s3 sync --no-sign-request",
             now, c12, du(c12), "tar OK; 3x4000 records",
             "4000x3 + Outcomes x3", f"{na}+{nb}+{nc}",
             "OK" if na == nb == nc == 4000 else "CHECK",
             "single-center MIMIC-II(BIDMC); a/b/c random splits -> negative control; ICUType 4 units"])

# WiDS 2020
w = f"{P}/data/external/wids2020/training_v2.csv"
if os.path.exists(w):
    rows.append(["WiDS2020_GOSSIS", "kaggle competitions download -c widsdatathon2020 (manual accept)",
                 "kaggle CLI (user) -> scp", now, w,
                 du(w), sha256_head(w)[:16] + "...",
                 "training_v2.csv (91713x186)", "1",
                 "OK", "147 hospitals, hospital_id+icu_id; static day-1; multi-site scale-up"])

# eICU demo
e = f"{P}/data/external/eicu_demo/patient.csv.gz"
if os.path.exists(e):
    rows.append(["eICU_CRD_demo", "physionet.org/files/eicu-crd-demo/2.0.1/ (main site down; local copy)",
                 "manual (physionet)", now, f"{P}/data/external/eicu_demo",
                 du(f"{P}/data/external/eicu_demo"), sha256_head(e)[:16] + "...",
                 "patient.csv.gz (+lab/hospital pending)", "1(partial)",
                 "PARTIAL", "186 hospitalid; descriptive figure only; rest待主站恢复"])

with open(MAN, "w") as f:
    f.write("\t".join(COLS) + "\n")
    for r in rows:
        f.write("\t".join(str(x) for x in r) + "\n")
print(open(MAN).read())
