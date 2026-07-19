import pandas as pd

COLUMN_NAMES = (
    ["engine_id", "cycle"]
    + [f"op_setting_{i}" for i in range(1, 4)]
    + [f"sensor_{i}" for i in range(1, 22)]
)


def _load_cmaps(path: str) -> pd.DataFrame:
    """
    Load a NASA C-MAPSS text file.
    """

    df = pd.read_csv(
        path,
        sep=r"\s+",
        header=None,
    )

    df = df.iloc[:, :26]
    df.columns = COLUMN_NAMES

    return df


def load_training_data(path: str) -> pd.DataFrame:
    """Load training dataset."""
    return _load_cmaps(path)


def load_test_data(path: str) -> pd.DataFrame:
    """Load test dataset."""
    return _load_cmaps(path)


def load_rul(path: str) -> pd.DataFrame:
    """Load test RUL labels."""

    rul = pd.read_csv(path, header=None)
    rul.columns = ["RUL"]

    return rul