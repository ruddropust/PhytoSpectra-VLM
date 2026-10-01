from pathlib import Path

from phytospectra.data import load_metadata
from phytospectra.splits import create_folds


def test_image_wise_five_fold_contract(tmp_path: Path):
    root = Path(__file__).resolve().parents[2] / "data/raw"
    frame = load_metadata(root)
    folds = create_folds(frame, tmp_path / "folds.csv", seed=2026)
    assert folds["ImageId"].nunique() == 1590
    assert folds.groupby("Fold").size().tolist() == [318] * 5

