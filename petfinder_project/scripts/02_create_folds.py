# scripts/02_create_folds.py

import pandas as pd
from sklearn.model_selection import StratifiedKFold
from pathlib import Path


def create_folds(
    data_path: str | Path,
    output_path: str | Path,
    n_splits: int = 5,
    random_state: int = 42,
) -> None:
    data_path = Path(data_path)
    output_path = Path(output_path)

    # Load train data
    df = pd.read_csv(data_path)

    # Basic checks
    required_cols = {"PetID", "AdoptionSpeed"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in {data_path}: {sorted(missing)}")

    if df["PetID"].duplicated().any():
        dup_n = int(df["PetID"].duplicated().sum())
        raise ValueError(
            f"Found {dup_n} duplicated PetID values in {data_path}. "
            "PetID must be unique per row for this folding approach."
        )

    # Create folds
    df["fold"] = -1
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    for fold_id, (_, val_idx) in enumerate(skf.split(X=df["PetID"], y=df["AdoptionSpeed"])):
        df.loc[val_idx, "fold"] = fold_id

    # Ensure all rows assigned
    if (df["fold"] < 0).any():
        raise RuntimeError("Some rows were not assigned to any fold. Check the split logic.")

    # Save minimal "contract" file
    output_path.parent.mkdir(parents=True, exist_ok=True)
    out_df = df[["PetID", "AdoptionSpeed", "fold"]].copy()
    out_df.to_csv(output_path, index=False)

    # Quick report
    print(f"Saved: {output_path} ({len(out_df)} rows)")
    print("\nRows per fold:")
    print(out_df["fold"].value_counts().sort_index())
    print("\nClass distribution per fold (counts):")
    print(pd.crosstab(out_df["fold"], out_df["AdoptionSpeed"]).sort_index())


if __name__ == "__main__":
    TRAIN_PATH = Path("data/raw/train.csv")
    OUT_PATH = Path("data/artifacts/train_folds.csv")

    create_folds(
        data_path=TRAIN_PATH,
        output_path=OUT_PATH,
        n_splits=5,
        random_state=42,
    )
