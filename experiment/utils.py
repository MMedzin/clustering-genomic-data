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
    SpectralClustering,
)
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import MinMaxScaler
from sklearn.base import TransformerMixin
from sklearn_extra.cluster import KMedoids
from somlearn.som import SOM
import scipy.io as sio
from itertools import product

FILE_DIR = Path(__file__).parent
DATASETS_PATH = FILE_DIR / "../datasets"

SEED = 23


# Data Loaders


def norm_data(
    data: pd.DataFrame, scaler: TransformerMixin = MinMaxScaler()
) -> pd.DataFrame:
    return pd.DataFrame(
        scaler.fit_transform(data), columns=data.columns, index=data.index
    )

    # GEMLER


def load_gemler_data_normed(scaler: TransformerMixin = MinMaxScaler()) -> Callable:
    def loader_func() -> tuple[pd.DataFrame, pd.Series]:
        data = pd.concat(
            [
                pd.DataFrame(
                    loadarff(DATASETS_PATH / "AP_Breast_Ovary.arff")[0]
                ).set_index("ID_REF"),
                pd.DataFrame(
                    loadarff(DATASETS_PATH / "AP_Colon_Kidney.arff")[0]
                ).set_index("ID_REF"),
                pd.DataFrame(
                    loadarff(DATASETS_PATH / "AP_Endometrium_Prostate.arff")[0]
                ).set_index("ID_REF"),
                pd.DataFrame(
                    loadarff(DATASETS_PATH / "AP_Omentum_Lung.arff")[0]
                ).set_index("ID_REF"),
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

        data = norm_data(data, scaler=scaler)

        return data, ground_truth.map(lambda x: x.decode("utf-8"))

    return loader_func


def load_gemler_normed_param_grid() -> list[dict]:
    """
    Param grid created with help of jupyter notebook: hyperparameters_exploration_gmler.ipynb
    """

    K_VALUES = np.arange(2, 15, 1)
    K_MEANS_INIT = ["k-means++", "random"]
    K_MEDOIDS_INIT = ["k-medoids++", "random"]
    LINKAGE_VALUES = ["ward", "complete", "average", "single"]
    BIRCH_THRESHOLD_VALUES = [
        2.5518847840970014,
        5.103769568194003,
        7.655654352291004,
        10.207539136388005,
        12.759423920485007,
    ]
    BIRCH_BRANCHING_FACTOR_VALUES = [5, 28, 52, 76, 100]
    EPS_VALUES = [
        10.810481126126229,
        11.42876822008078,
        11.24727069231537,
        11.659056346150447,
        11.455016452632346,
        11.189865257207094,
        11.661372548192299,
        11.715841369400247,
        11.758441006716266,
        11.819918622368519,
    ]
    MIN_SAMPLES_VALUES = [3, 6, 10, 14, 18, 22, 26, 30, 34, 38]
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

    return {
        "KMeans": {
            "cluster_algo": [KMeans(n_init="auto", random_state=SEED)],
            "cluster_algo__n_clusters": K_VALUES,
            "cluster_algo__init": K_MEANS_INIT,
        },
        "KMedoids": {
            "cluster_algo": [KMedoids(random_state=SEED)],
            "cluster_algo__n_clusters": K_VALUES,
            "cluster_algo__init": K_MEDOIDS_INIT,
        },
        "AgglomerativeClustering": {
            "cluster_algo": [AgglomerativeClustering()],
            "cluster_algo__n_clusters": K_VALUES,
            "cluster_algo__linkage": LINKAGE_VALUES,
        },
        # "Birch": {
        #     "cluster_algo": [Birch()],
        #     "cluster_algo__threshold": BIRCH_THRESHOLD_VALUES,
        #     "cluster_algo__branching_factor": BIRCH_BRANCHING_FACTOR_VALUES,
        #     "cluster_algo__n_clusters": list(K_VALUES) + [None],
        # },
        # "DBSCAN": [
        #     {
        #         "cluster_algo": [DBSCAN()],
        #         "cluster_algo__eps": [eps],
        #         "cluster_algo__min_samples": [min_samples],
        #     }
        #     for eps, min_samples in zip(EPS_VALUES, MIN_SAMPLES_VALUES)
        # ],
        # "OPTICS": {
        #     "cluster_algo": [OPTICS(cluster_method="dbscan")],
        #     "cluster_algo__min_samples": MIN_SAMPLES_VALUES,
        # },
        # "GaussianMixture": {
        #     "cluster_algo": [GaussianMixture(random_state=SEED)],
        #     "cluster_algo__n_components": K_VALUES,
        #     "cluster_algo__covariance_type": COVARIANCE_TYPE_VALUES,
        # },
        # "SOM": [
        #     {
        #         "cluster_algo": [SOM(random_state=SEED)],
        #         "cluster_algo__n_columns": [k1],
        #         "cluster_algo__n_rows": [k2],
        #         "cluster_algo__initialcodebook": SOM_INITIALCODEBOOK_VALUES,
        #         "cluster_algo__neighborhood": SOM_NEIGHBORHOOD_VALUES,
        #     }
        #     for k1, k2 in product([1] + list(K_VALUES), list(K_VALUES))
        #     if k1 * k2 <= max(K_VALUES)
        # ],
        "AffinityPropagation": {
            "cluster_algo": [AffinityPropagation(random_state=SEED)],
            "cluster_algo__damping": AFFINITY_PROP_DUMPING_VALUES,
            "cluster_algo__preference": AFFINITY_PROP_PREFERENCE_VALUES,
        },
        "SpectralClustering": [
            {
                "cluster_algo": [SpectralClustering(random_state=SEED)],
                "cluster_algo__n_clusters": K_VALUES,
                "cluster_algo__affinity": ["rbf"],
                "cluster_algo__assign_labels": ["kmeans", "discretize", "cluster_qr"],
            },
            {
                "cluster_algo": [SpectralClustering(random_state=SEED)],
                "cluster_algo__n_clusters": K_VALUES,
                "cluster_algo__affinity": ["nearest_neighbors"],
                "cluster_algo__n_neighbors": MIN_SAMPLES_VALUES,
                "cluster_algo__assign_labels": ["kmeans", "discretize", "cluster_qr"],
            },
        ],
    }


# METABRIC
def load_metabric_data_normed(
    scaler: TransformerMixin = MinMaxScaler(),
) -> Callable:
    def loader_func() -> tuple[pd.DataFrame, pd.Series]:
        data_mat = sio.loadmat(DATASETS_PATH / "BRCA1View20000.mat")
        data = pd.DataFrame(data_mat["data"]).transpose()
        data.index = data_mat["id"][0]
        data.columns = map(lambda x: x[0], data_mat["gene"][0])
        ground_truth = pd.Series(data_mat["targets"][:, 0], index=data.index)

        data = norm_data(data, scaler=scaler)

        return data, ground_truth

    return loader_func


def load_metabric_normed_param_grid() -> list[dict]:
    """
    Param grid created with help of jupyter notebook: hyperparameters_exploration_metabric.ipynb
    """

    K_VALUES = np.arange(2, 21, 1)
    K_MEANS_INIT = ["k-means++", "random"]
    K_MEDOIDS_INIT = ["k-medoids++", "random"]
    LINKAGE_VALUES = ["ward", "complete", "average", "single"]
    BIRCH_THRESHOLD_VALUES = [
        5.114437120006565,
        10.22887424001313,
        15.343311360019696,
        20.45774848002626,
        25.572185600032828,
    ]
    BIRCH_BRANCHING_FACTOR_VALUES = [5, 28, 52, 76, 100]
    EPS_VALUES = [
        20.69517473682061,
        21.424819891686553,
        21.65635555044067,
        21.943242935651888,
        22.04489489265609,
        22.053542447540618,
        22.23324254655041,
        22.33930803401733,
        22.522489554319662,
        22.621118201430036,
    ]
    MIN_SAMPLES_VALUES = [5, 10, 15, 21, 26, 31, 37, 42, 47, 53]
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

    return {
        "KMeans": {
            "cluster_algo": [KMeans(n_init="auto", random_state=SEED)],
            "cluster_algo__n_clusters": K_VALUES,
            "cluster_algo__init": K_MEANS_INIT,
        },
        "KMedoids": {
            "cluster_algo": [KMedoids(random_state=SEED)],
            "cluster_algo__n_clusters": K_VALUES,
            "cluster_algo__init": K_MEDOIDS_INIT,
        },
        "AgglomerativeClustering": {
            "cluster_algo": [AgglomerativeClustering()],
            "cluster_algo__n_clusters": K_VALUES,
            "cluster_algo__linkage": LINKAGE_VALUES,
        },
        # "Birch": {
        #     "cluster_algo": [Birch()],
        #     "cluster_algo__threshold": BIRCH_THRESHOLD_VALUES,
        #     "cluster_algo__branching_factor": BIRCH_BRANCHING_FACTOR_VALUES,
        #     "cluster_algo__n_clusters": list(K_VALUES) + [None],
        # },
        # "DBSCAN": [
        #     {
        #         "cluster_algo": [DBSCAN()],
        #         "cluster_algo__eps": [eps],
        #         "cluster_algo__min_samples": [min_samples],
        #     }
        #     for eps, min_samples in zip(EPS_VALUES, MIN_SAMPLES_VALUES)
        # ],
        # "OPTICS": {
        #     "cluster_algo": [OPTICS(cluster_method="dbscan")],
        #     "cluster_algo__min_samples": MIN_SAMPLES_VALUES,
        # },
        # "GaussianMixture": {
        #     "cluster_algo": [GaussianMixture(random_state=SEED)],
        #     "cluster_algo__n_components": K_VALUES,
        #     "cluster_algo__covariance_type": COVARIANCE_TYPE_VALUES,
        # },
        # "SOM": [
        #     {
        #         "cluster_algo": [SOM(random_state=SEED)],
        #         "cluster_algo__n_columns": [k1],
        #         "cluster_algo__n_rows": [k2],
        #         "cluster_algo__initialcodebook": SOM_INITIALCODEBOOK_VALUES,
        #         "cluster_algo__neighborhood": SOM_NEIGHBORHOOD_VALUES,
        #     }
        #     for k1, k2 in product([1] + list(K_VALUES), list(K_VALUES))
        #     if k1 * k2 <= max(K_VALUES)
        # ],
        "AffinityPropagation": {
            "cluster_algo": [AffinityPropagation(random_state=SEED)],
            "cluster_algo__damping": AFFINITY_PROP_DUMPING_VALUES,
            "cluster_algo__preference": AFFINITY_PROP_PREFERENCE_VALUES,
        },
        "SpectralClustering": [
            {
                "cluster_algo": [SpectralClustering(random_state=SEED)],
                "cluster_algo__n_clusters": K_VALUES,
                "cluster_algo__affinity": ["rbf"],
                "cluster_algo__assign_labels": ["kmeans", "discretize", "cluster_qr"],
            },
            {
                "cluster_algo": [SpectralClustering(random_state=SEED)],
                "cluster_algo__n_clusters": K_VALUES,
                "cluster_algo__affinity": ["nearest_neighbors"],
                "cluster_algo__n_neighbors": MIN_SAMPLES_VALUES,
                "cluster_algo__assign_labels": ["kmeans", "discretize", "cluster_qr"],
            },
        ],
    }


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
