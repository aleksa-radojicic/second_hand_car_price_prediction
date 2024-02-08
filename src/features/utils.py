import pandas as pd

from src.config import DTYPE_BACKEND


def downcast_nonnumerical_dtypes(df, binary, ordinal, nominal):
    # If backend is "numpy_nullable"
    backend = ""

    if DTYPE_BACKEND == "pyarrow":
        backend = f"[{DTYPE_BACKEND}]"

    df = df.copy()

    for c in binary:
        df[c] = (
            df.loc[:, c]
            .apply(lambda x: True if x == "Y" else False)
            .astype(f"bool{backend}")
        )

    for c in ordinal:
        df[c] = pd.Categorical(df.loc[:, c], ordered=True)

    for c in nominal:
        df[c] = pd.Categorical(df.loc[:, c], ordered=False)

    return df
