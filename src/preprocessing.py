import pandas as pd


def add_rul(
    df: pd.DataFrame,
    max_rul: int = 125,
) -> pd.DataFrame:
    """
    Compute Remaining Useful Life (RUL)
    for every engine.

    Parameters
    ----------
    df : pd.DataFrame
        NASA C-MAPSS training data.

    max_rul : int
        Maximum RUL cap.

    Returns
    -------
    pd.DataFrame
    """

    required = {"engine_id", "cycle"}

    missing = required - set(df.columns)

    if missing:
        raise ValueError(
            f"Missing columns: {missing}"
        )

    max_cycle = (
        df.groupby("engine_id")["cycle"]
        .max()
        .rename("max_cycle")
    )

    df = df.merge(
        max_cycle,
        on="engine_id",
    )

    df["RUL"] = (
        df["max_cycle"]
        - df["cycle"]
    )

    df["RUL"] = df["RUL"].clip(
        upper=max_rul
    )

    df = df.drop(
        columns="max_cycle"
    )

    return df