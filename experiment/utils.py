import pandas as pd
import numpy as np
from pathlib import Path
from scipy.io.arff import loadarff
from sklearn.base import BaseEstimator
from typing import Union, Callable, Any

from sklearn.cluster import (
    DBSCAN,
    OPTICS,
    AffinityPropagation,
    AgglomerativeClustering,
    Birch,
    KMeans,
)
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import MinMaxScaler
from sklearn_extra.cluster import KMedoids
from somlearn.som import SOM
import scipy.io as sio

FILE_DIR = Path(__file__).parent
DATASETS_PATH = FILE_DIR / "../datasets"

SEED = 23


# Data Loaders


def min_max_norm_data(data: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        MinMaxScaler().fit_transform(data), columns=data.columns, index=data.index
    )


# GEMLER
def load_gemler_data_normed() -> tuple[pd.DataFrame, pd.Series]:
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

    ground_truth = data.loc[:, "Tissue"]
    data = data.drop(columns="Tissue")

    data = min_max_norm_data(data)

    return data, ground_truth


def load_gemler_normed_param_grid() -> list[dict]:
    """
    Param grid created with help of jupyter notebook: hyperparameters_exploration_gmler.ipynb
    """

    K_VALUES = [2, 3, 4, 5, 6, 7, 8, 9, 10, 15]
    K_MEANS_INIT = ["k-means++", "random"]
    K_MEDOIDS_INIT = ["k-medoids++", "random"]
    LINKAGE_VALUES = ["ward", "complete", "average", "single"]
    BIRCH_THRESHOLD_VALUES = [
        2.5518847840970014,
        5.103769568194003,
        7.655654352291004,
        10.207539136388005,
        12.759423920485007,
        15.311308704582007,
        17.86319348867901,
        20.41507827277601,
        22.966963056873013,
        25.518847840970015,
        None,
    ]
    BIRCH_BRANCHING_FACTOR_VALUES = [5, 28, 52, 76, 100]
    EPS_VALUES = [11.5, 12.75, 13.5, 14.0, 14.5]
    MIN_SAMPLES_VALUES = [5, 90, 175, 260, 346, 431, 516, 601, 687, 772]
    COVARIANCE_TYPE_VALUES = ["full", "tied", "diag", "spherical"]
    SOM_INITIALCODEBOOK_VALUES = [None, "pca"]
    SOM_NEIGHBORHOOD_VALUES = ["gaussian", "bubble"]
    AFFINITY_PROP_DUMPING_VALUES = np.arange(0.5, 1, 0.1)
    AFFINITY_PROP_PREFERENCE_VALUES = [
        -310.54188853512886,
        -276.0372342534479,
        -241.5325799717669,
        -207.0279256900859,
        -172.52327140840492,
        -138.01861712672394,
        -103.51396284504295,
        -69.00930856336197,
        -34.504654281680985,
        0.0,
    ]

    return [
        {
            "cluster_algo": [KMeans(n_init="auto", random_state=SEED)],
            "cluster_algo__n_clusters": K_VALUES,
            "cluster_algo__init": K_MEANS_INIT,
        },
        {
            "cluster_algo": [KMedoids(random_state=SEED)],
            "cluster_algo__n_clusters": K_VALUES,
            "cluster_algo__init": K_MEDOIDS_INIT,
        },
        {
            "cluster_algo": [AgglomerativeClustering()],
            "cluster_algo__n_clusters": K_VALUES,
            "cluster_algo__linkage": LINKAGE_VALUES,
        },
        {
            "cluster_algo": [Birch()],
            "cluster_algo__threshold": BIRCH_THRESHOLD_VALUES,
            "cluster_algo__branching_factor": BIRCH_BRANCHING_FACTOR_VALUES,
            "cluster_algo__n_clusters": K_VALUES + [None],
        },
        {
            "cluster_algo": [DBSCAN()],
            "cluster_algo__eps": EPS_VALUES,
            "cluster_algo__min_samples": MIN_SAMPLES_VALUES,
        },
        {
            "cluster_algo": [OPTICS(cluster_method="dbscan")],
            "cluster_algo__min_samples": MIN_SAMPLES_VALUES,
        },
        {
            "cluster_algo": [GaussianMixture(random_state=SEED)],
            "cluster_algo__n_components": K_VALUES,
            "cluster_algo__covariance_type": COVARIANCE_TYPE_VALUES,
        },
        {
            "cluster_algo": [SOM(random_state=SEED)],
            "cluster_algo__n_cols": [1] + K_VALUES[:-1],
            "cluster_algo__n_rows": K_VALUES,
            "cluster_algo__initialcodebook": SOM_INITIALCODEBOOK_VALUES,
            "cluster_algo__neighborhood": SOM_NEIGHBORHOOD_VALUES,
        },
        {
            "cluster_algo": [SOM(random_state=SEED)],
            "cluster_algo__n_cols": K_VALUES,
            "cluster_algo__n_rows": [1] + K_VALUES[:-1],
            "cluster_algo__initialcodebook": SOM_INITIALCODEBOOK_VALUES,
            "cluster_algo__neighborhood": SOM_NEIGHBORHOOD_VALUES,
        },
        {
            "cluster_algo": [AffinityPropagation(random_state=SEED)],
            "cluster_algo__damping": AFFINITY_PROP_DUMPING_VALUES,
            "cluster_algo__preference": AFFINITY_PROP_PREFERENCE_VALUES,
        },
    ]


# METABRIC
def load_metabric_data_normed() -> tuple[pd.DataFrame, pd.Series]:
    data_mat = sio.loadmat(DATASETS_PATH / "BRCA1View20000.mat")
    data = pd.DataFrame(data_mat["data"]).transpose()
    data.index = data_mat["id"][0]
    data.columns = map(lambda x: x[0], data_mat["gene"][0])
    ground_truth = pd.Series(data_mat["targets"][:, 0], index=data.index)

    data = min_max_norm_data(data)

    return data, ground_truth


def load_metabric_normed_param_grid() -> list[dict]:
    """
    Param grid created with help of jupyter notebook: hyperparameters_exploration_metabric.ipynb
    """

    K_VALUES = [2, 3, 4, 5, 6, 7, 8, 9, 10, 15]
    K_MEANS_INIT = ["k-means++", "random"]
    K_MEDOIDS_INIT = ["k-medoids++", "random"]
    LINKAGE_VALUES = ["ward", "complete", "average", "single"]
    BIRCH_THRESHOLD_VALUES = [
        5.114437120006565,
        10.22887424001313,
        15.343311360019696,
        20.45774848002626,
        25.572185600032824,
        30.68662272003939,
        35.80105984004596,
        40.91549696005252,
        46.029934080059085,
        51.144371200065656,
        None,
    ]
    BIRCH_BRANCHING_FACTOR_VALUES = [5, 28, 52, 76, 100]
    EPS_VALUES = [22.5, 24.5, 25.0, 26.0, 27.0]
    MIN_SAMPLES_VALUES = [5, 122, 240, 358, 476, 594, 712, 830, 948, 1066]
    COVARIANCE_TYPE_VALUES = ["full", "tied", "diag", "spherical"]
    SOM_INITIALCODEBOOK_VALUES = [None, "pca"]
    SOM_NEIGHBORHOOD_VALUES = ["gaussian", "bubble"]
    AFFINITY_PROP_DUMPING_VALUES = np.arange(0.5, 1, 0.1)
    AFFINITY_PROP_PREFERENCE_VALUES = [
        -1076.453721835283,
        -956.8477527424737,
        -837.2417836496645,
        -717.6358145568554,
        -598.029845464046,
        -478.42387637123693,
        -358.8179072784277,
        -239.21193818561846,
        -119.60596909280923,
        0.0,
    ]

    return [
        {
            "cluster_algo": [KMeans(n_init="auto", random_state=SEED)],
            "cluster_algo__n_clusters": K_VALUES,
            "cluster_algo__init": K_MEANS_INIT,
        },
        {
            "cluster_algo": [KMedoids(random_state=SEED)],
            "cluster_algo__n_clusters": K_VALUES,
            "cluster_algo__init": K_MEDOIDS_INIT,
        },
        {
            "cluster_algo": [AgglomerativeClustering()],
            "cluster_algo__n_clusters": K_VALUES,
            "cluster_algo__linkage": LINKAGE_VALUES,
        },
        {
            "cluster_algo": [Birch()],
            "cluster_algo__threshold": BIRCH_THRESHOLD_VALUES,
            "cluster_algo__branching_factor": BIRCH_BRANCHING_FACTOR_VALUES,
            "cluster_algo__n_clusters": K_VALUES + [None],
        },
        {
            "cluster_algo": [DBSCAN()],
            "cluster_algo__eps": EPS_VALUES,
            "cluster_algo__min_samples": MIN_SAMPLES_VALUES,
        },
        {
            "cluster_algo": [OPTICS(cluster_method="dbscan")],
            "cluster_algo__min_samples": MIN_SAMPLES_VALUES,
        },
        {
            "cluster_algo": [GaussianMixture(random_state=SEED)],
            "cluster_algo__n_components": K_VALUES,
            "cluster_algo__covariance_type": COVARIANCE_TYPE_VALUES,
        },
        {
            "cluster_algo": [SOM(random_state=SEED)],
            "cluster_algo__n_cols": [1] + K_VALUES[:-1],
            "cluster_algo__n_rows": K_VALUES,
            "cluster_algo__initialcodebook": SOM_INITIALCODEBOOK_VALUES,
            "cluster_algo__neighborhood": SOM_NEIGHBORHOOD_VALUES,
        },
        {
            "cluster_algo": [SOM(random_state=SEED)],
            "cluster_algo__n_cols": K_VALUES,
            "cluster_algo__n_rows": [1] + K_VALUES[:-1],
            "cluster_algo__initialcodebook": SOM_INITIALCODEBOOK_VALUES,
            "cluster_algo__neighborhood": SOM_NEIGHBORHOOD_VALUES,
        },
        {
            "cluster_algo": [AffinityPropagation(random_state=SEED)],
            "cluster_algo__damping": AFFINITY_PROP_DUMPING_VALUES,
            "cluster_algo__preference": AFFINITY_PROP_PREFERENCE_VALUES,
        },
    ]


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
