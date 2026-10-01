"""Leakage-safe image-wise 5-fold split generation."""

from __future__ import annotations

from pathlib import Path
import hashlib

import pandas as pd
from sklearn.model_selection import StratifiedKFold


def create_folds(frame: pd.DataFrame, output: str | Path, seed: int = 2026) -> pd.DataFrame:
    result = frame.copy()
    strata = result["Cultivar"].astype(str) + "__F" + result["Fertilizer"].astype(str)
    result["Stratum"] = strata
    result["Fold"] = -1
    splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=int(seed))
    for fold, (_, validation) in enumerate(splitter.split(result["ImageId"], strata)):
        result.loc[result.index[validation], "Fold"] = fold
    if sorted(result.groupby("Fold").size().tolist()) != [318] * 5:
        raise RuntimeError("Five-fold sizes are not 318 images each")
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output, index=False)
    digest = hashlib.sha256(output.read_bytes()).hexdigest()
    return result


def split_fold(frame: pd.DataFrame, fold: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    train = frame.loc[frame["Fold"] != int(fold)].copy()
    validation = frame.loc[frame["Fold"] == int(fold)].copy()
    overlap = set(train["ImageId"]) & set(validation["ImageId"])
    if overlap or len(train) != 1272 or len(validation) != 318:
        raise RuntimeError("Invalid image-wise outer split")
    return train, validation

