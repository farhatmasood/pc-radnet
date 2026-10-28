"""Mean reference disc height over the training split (constant-predictor baseline)."""
from pathlib import Path

import pandas as pd
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from repo_config import load_config  # noqa: E402

DS = Path(load_config()["paths"]["dataset_root"])
SPLIT = Path(__file__).resolve().parents[1] / "data" / "train.txt"


def train_mean():
    pids = {int(Path(l.strip()).stem.split("_")[2]) for l in open(SPLIT) if l.strip()}
    df = pd.read_csv(DS / "IVDHeights.csv")
    df = df[df.Patient_ID.isin(pids)]
    return float(pd.concat([df.D3_Ht, df.D4_Ht, df.D5_Ht]).mean())
