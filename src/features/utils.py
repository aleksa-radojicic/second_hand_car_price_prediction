import pandas as pd
from sklearn.feature_selection import f_regression


def get_anova_importance_scores(X: pd.DataFrame, y: pd.DataFrame) -> pd.DataFrame:
    f_stat, p_val = f_regression(X, y)

    # Fit the feature selector and sort the results by score
    scores = pd.DataFrame(
        {"scores": f_stat, "p_val": p_val}, index=X.columns
    ).sort_values(by="scores", ascending=False)

    scores.index.name = "anova_importance_scores"

    return scores
