import pandas as pd
import numpy as np
from pathlib import Path
from scipy.io.arff import loadarff
from sklearn.base import BaseEstimator
from typing import Union, Callable, Any

FILE_DIR = Path(__file__).parent
DATASETS_PATH = FILE_DIR / "../datasets"

# Data Loaders


def load_gemler_data() -> tuple[pd.DataFrame, pd.Series]:
    data = pd.concat(
        [
            pd.DataFrame(loadarff(DATASETS_PATH / "AP_Breast_Ovary.arff")[0]).set_index(
                "ID_REF"
            ),
            pd.DataFrame(loadarff(DATASETS_PATH / "AP_Colon_Kidney.arff")[0]).set_index(
                "ID_REF"
            ),
            pd.DataFrame(
                loadarff(DATASETS_PATH / "AP_Endometrium_Prostate.arff")[0]
            ).set_index("ID_REF"),
            pd.DataFrame(loadarff(DATASETS_PATH / "AP_Omentum_Lung.arff")[0]).set_index(
                "ID_REF"
            ),
            pd.DataFrame(
                loadarff(DATASETS_PATH / "AP_Prostate_Uterus.arff")[0]
            ).set_index("ID_REF"),
        ]
    )
    data = data.dropna(
        how="any", axis="columns"
    )  # Keep only genes common in all datasets
    data = data.loc[~data.index.duplicated()]  # Drop duplicates of Prostate samples
    return data.drop(columns="Tissue"), data.loc[:, "Tissue"]


# Scorer factories


def make_clustering_scorer_unsupervised(score_func: Callable) -> Callable:
    def scorer(
        estimator: BaseEstimator, X: Union[pd.DataFrame, np.array], _y: Any
    ) -> float:
        labels = estimator.fit_predict(X)
        return score_func(X, labels)

    return scorer


def make_clustering_scorer_supervised(score_func: Callable) -> Callable:
    def scorer(
        estimator: BaseEstimator,
        X: Union[pd.DataFrame, np.array],
        y: Union[pd.Series, np.array],
    ) -> float:
        labels = estimator.fit_predict(X)
        return score_func(y, labels)

    return scorer
