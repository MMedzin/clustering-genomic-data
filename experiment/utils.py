from itertools import product
from pathlib import Path
from typing import Any, Callable, Optional, Union

import numpy as np
import pandas as pd
import scipy.io as sio
from scipy.io.arff import loadarff
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.cluster import (
    DBSCAN,
    OPTICS,
    AffinityPropagation,
    AgglomerativeClustering,
    Birch,
    KMeans,
    SpectralClustering,
)
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import FunctionTransformer, MinMaxScaler
from sklearn_extra.cluster import KMedoids
from somlearn.som import SOM

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


def noop_func(
    X: Union[pd.DataFrame, np.array], y: Optional[Any] = None
) -> Union[pd.DataFrame, np.array]:
    return X


NOOP_TRANSFORMER = FunctionTransformer(noop_func)


def remove_outliers(
    data: pd.DataFrame,
    ground_truth: pd.Series,
    model: Optional[BaseEstimator] = LocalOutlierFactor(n_neighbors=2),
) -> pd.DataFrame:
    if model is None:
        return data, ground_truth
    pred = pd.Series(model.fit_predict(data), index=data.index)
    return data.loc[pred == 1], ground_truth.loc[pred == 1]


# GEMLER


def load_gemler_data_normed(
    scaler: TransformerMixin = MinMaxScaler(),
    outliers_model: Optional[BaseEstimator] = LocalOutlierFactor(n_neighbors=2),
) -> Callable:
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

        data, ground_truth = remove_outliers(data, ground_truth, outliers_model)
        data = norm_data(data, scaler=scaler)

        return data, ground_truth.map(lambda x: x.decode("utf-8"))

    return loader_func


def load_gemler_normed_param_grid() -> list[dict]:
    """
    Param grid created with help of jupyter notebook: hyperparameters_exploration_gmler.ipynb
    """

    K_VALUES = np.arange(2, 14, 1)
    K_MEANS_INIT = ["k-means++", "random"]
    K_MEDOIDS_INIT = ["k-medoids++", "random"]
    LINKAGE_VALUES = ["ward", "complete", "average", "single"]
    BIRCH_THRESHOLD_VALUES = [
        1.3214501443830946,
        2.642900288766189,
        3.964350433149284,
        5.285800577532378,
        6.607250721915473,
    ]
    BIRCH_BRANCHING_FACTOR_VALUES = [5, 28, 52, 76, 100]
    EPS_VALUES = [
        5.5634014878010625,
        10.949553505487414,
        11.280439444443894,
        11.527275168957221,
        11.521296693873124,
    ]
    MIN_SAMPLES_VALUES = [3, 11, 20, 28, 37]
    COVARIANCE_TYPE_VALUES = ["full", "tied", "diag", "spherical"]
    SOM_INITIALCODEBOOK_VALUES = [None, "pca"]
    SOM_NEIGHBORHOOD_VALUES = ["gaussian", "bubble"]
    AFFINITY_PROP_DUMPING_VALUES = np.arange(0.5, 1, 0.1)
    AFFINITY_PROP_PREFERENCE_VALUES = [
        -324.33104770458385,
        -288.29426462629675,
        -252.25748154800965,
        -216.22069846972258,
        -180.18391539143548,
        -144.14713231314838,
        -108.1103492348613,
        -72.0735661565742,
        -36.0367830782871,
        0.0,
    ]
    PCA_COMPONENTS = 199

    return {
        # "KMeans": {
        #     # "reduce_dim": ["passthrough", PCA(n_components=PCA_COMPONENTS)],
        #     "reduce_dim": [PCA(n_components=PCA_COMPONENTS)],
        #     "cluster_algo": [KMeans(n_init="auto", random_state=SEED)],
        #     "cluster_algo__n_clusters": K_VALUES,
        #     "cluster_algo__init": K_MEANS_INIT,
        # },
        # "KMedoids": {
        #     # "reduce_dim": ["passthrough", PCA(n_components=PCA_COMPONENTS)],
        #     "reduce_dim": [PCA(n_components=PCA_COMPONENTS)],
        #     "cluster_algo": [KMedoids(random_state=SEED)],
        #     "cluster_algo__n_clusters": K_VALUES,
        #     "cluster_algo__init": K_MEDOIDS_INIT,
        # },
        # "AgglomerativeClustering": {
        #     # "reduce_dim": ["passthrough", PCA(n_components=PCA_COMPONENTS)],
        #     "reduce_dim": [PCA(n_components=PCA_COMPONENTS)],
        #     "cluster_algo": [AgglomerativeClustering()],
        #     "cluster_algo__n_clusters": K_VALUES,
        #     "cluster_algo__linkage": LINKAGE_VALUES,
        # },
        # "Birch": {
        #     # "reduce_dim": ["passthrough", PCA(n_components=PCA_COMPONENTS)],
        #     "reduce_dim": [PCA(n_components=PCA_COMPONENTS)],
        #     "cluster_algo": [Birch()],
        #     "cluster_algo__threshold": BIRCH_THRESHOLD_VALUES,
        #     "cluster_algo__branching_factor": BIRCH_BRANCHING_FACTOR_VALUES,
        #     "cluster_algo__n_clusters": list(K_VALUES) + [None],
        # },
        # "DBSCAN": [
        #     {
        #         # "reduce_dim": ["passthrough", PCA(n_components=PCA_COMPONENTS)],
        #         "reduce_dim": [PCA(n_components=PCA_COMPONENTS)],
        #         "cluster_algo": [DBSCAN()],
        #         "cluster_algo__eps": [eps],
        #         "cluster_algo__min_samples": [min_samples],
        #     }
        #     for eps, min_samples in zip(EPS_VALUES, MIN_SAMPLES_VALUES)
        # ],
        # "OPTICS": {
        #     # "reduce_dim": ["passthrough", PCA(n_components=PCA_COMPONENTS)],
        #     "reduce_dim": [PCA(n_components=PCA_COMPONENTS)],
        #     "cluster_algo": [OPTICS(cluster_method="dbscan")],
        #     "cluster_algo__min_samples": MIN_SAMPLES_VALUES,
        # },
        # "GaussianMixture": {
        #     "reduce_dim": [PCA(n_components=PCA_COMPONENTS)],
        #     "cluster_algo": [GaussianMixture(random_state=SEED)],
        #     "cluster_algo__n_components": K_VALUES,
        #     "cluster_algo__covariance_type": COVARIANCE_TYPE_VALUES,
        # },
        "SOM": [
            {
                # "reduce_dim": ["passthrough", PCA(n_components=PCA_COMPONENTS)],
                "reduce_dim": [PCA(n_components=PCA_COMPONENTS)],
                "cluster_algo": [SOM(random_state=SEED)],
                "cluster_algo__n_columns": [k1],
                "cluster_algo__n_rows": [k2],
                "cluster_algo__initialcodebook": SOM_INITIALCODEBOOK_VALUES,
                "cluster_algo__neighborhood": SOM_NEIGHBORHOOD_VALUES,
            }
            for k1, k2 in [(2, 1)]
            # for k1, k2 in product([1] + list(K_VALUES), list(K_VALUES))
            # if k1 * k2 <= max(K_VALUES)
        ],
        # "AffinityPropagation": {
        #     # "reduce_dim": ["passthrough", PCA(n_components=PCA_COMPONENTS)],
        #     "reduce_dim": [PCA(n_components=PCA_COMPONENTS)],
        #     "cluster_algo": [AffinityPropagation(random_state=SEED)],
        #     "cluster_algo__damping": AFFINITY_PROP_DUMPING_VALUES,
        #     "cluster_algo__preference": AFFINITY_PROP_PREFERENCE_VALUES,
        # },
        # "SpectralClustering": [
        #     {
        #         # "reduce_dim": ["passthrough", PCA(n_components=PCA_COMPONENTS)],
        #         "reduce_dim": [PCA(n_components=PCA_COMPONENTS)],
        #         "cluster_algo": [SpectralClustering(random_state=SEED)],
        #         "cluster_algo__n_clusters": K_VALUES,
        #         "cluster_algo__affinity": ["rbf"],
        #         "cluster_algo__assign_labels": ["kmeans", "discretize", "cluster_qr"],
        #     },
        #     {
        #         # "reduce_dim": ["passthrough", PCA(n_components=PCA_COMPONENTS)],
        #         "reduce_dim": [PCA(n_components=PCA_COMPONENTS)],
        #         "cluster_algo": [SpectralClustering(random_state=SEED)],
        #         "cluster_algo__n_clusters": K_VALUES,
        #         "cluster_algo__affinity": ["nearest_neighbors"],
        #         "cluster_algo__n_neighbors": MIN_SAMPLES_VALUES,
        #         "cluster_algo__assign_labels": ["kmeans", "discretize", "cluster_qr"],
        #     },
        # ],
    }


# METABRIC
def load_metabric_data_normed(
    scaler: TransformerMixin = MinMaxScaler(),
    outliers_model: Optional[BaseEstimator] = IsolationForest(
        max_features=0.01, random_state=SEED
    ),
) -> Callable:
    def loader_func() -> tuple[pd.DataFrame, pd.Series]:
        data_mat = sio.loadmat(DATASETS_PATH / "BRCA1View20000.mat")
        data = pd.DataFrame(data_mat["data"]).transpose()
        data.index = data_mat["id"][0]
        data.columns = map(lambda x: x[0], data_mat["gene"][0])
        ground_truth = pd.Series(data_mat["targets"][:, 0], index=data.index)

        data, ground_truth = remove_outliers(data, ground_truth, outliers_model)
        data = norm_data(data, scaler=scaler)

        return data, ground_truth

    return loader_func


def load_metabric_normed_param_grid() -> list[dict]:
    """
    Param grid created with help of jupyter notebook: hyperparameters_exploration_metabric.ipynb
    """

    K_VALUES = np.arange(2, 19, 1)
    K_MEANS_INIT = ["k-means++", "random"]
    K_MEDOIDS_INIT = ["k-medoids++", "random"]
    LINKAGE_VALUES = ["ward", "complete", "average", "single"]
    BIRCH_THRESHOLD_VALUES = [
        2.283273971400119,
        4.566547942800238,
        6.849821914200357,
        9.133095885600476,
        11.416369857000594,
    ]
    BIRCH_BRANCHING_FACTOR_VALUES = [5, 28, 52, 76, 100]
    EPS_VALUES = [
        20.508388381488942,
        21.09282513466498,
        21.815374792658716,
        22.133380156060504,
        22.285870736598028,
    ]
    MIN_SAMPLES_VALUES = [5, 16, 28, 40, 52]
    COVARIANCE_TYPE_VALUES = ["full", "tied", "diag", "spherical"]
    SOM_INITIALCODEBOOK_VALUES = [None, "pca"]
    SOM_NEIGHBORHOOD_VALUES = ["gaussian", "bubble"]
    AFFINITY_PROP_DUMPING_VALUES = np.arange(0.5, 1, 0.1)
    [
        -1133.0037800747316,
        -1007.1144711775391,
        -881.2251622803468,
        -755.3358533831545,
        -629.446544485962,
        -503.5572355887696,
        -377.66792669157724,
        -251.7786177943849,
        -125.88930889719245,
        0.0,
    ]
    AFFINITY_PROP_PREFERENCE_VALUES = [
        -1133.0037800747316,
        -1007.1144711775391,
        -881.2251622803468,
        -755.3358533831545,
        -629.446544485962,
        -503.5572355887696,
        -377.66792669157724,
        -251.7786177943849,
        -125.88930889719245,
        0.0,
    ]
    PCA_COMPONENTS = 242

    return {
        "KMeans": {
            "reduce_dim": ["passthrough", PCA(n_components=PCA_COMPONENTS)],
            "cluster_algo": [KMeans(n_init="auto", random_state=SEED)],
            "cluster_algo__n_clusters": K_VALUES,
            "cluster_algo__init": K_MEANS_INIT,
        },
        "KMedoids": {
            "reduce_dim": ["passthrough", PCA(n_components=PCA_COMPONENTS)],
            "cluster_algo": [KMedoids(random_state=SEED)],
            "cluster_algo__n_clusters": K_VALUES,
            "cluster_algo__init": K_MEDOIDS_INIT,
        },
        "AgglomerativeClustering": {
            "reduce_dim": ["passthrough", PCA(n_components=PCA_COMPONENTS)],
            "cluster_algo": [AgglomerativeClustering()],
            "cluster_algo__n_clusters": K_VALUES,
            "cluster_algo__linkage": LINKAGE_VALUES,
        },
        "Birch": {
            "reduce_dim": ["passthrough", PCA(n_components=PCA_COMPONENTS)],
            "cluster_algo": [Birch()],
            "cluster_algo__threshold": BIRCH_THRESHOLD_VALUES,
            "cluster_algo__branching_factor": BIRCH_BRANCHING_FACTOR_VALUES,
            "cluster_algo__n_clusters": list(K_VALUES) + [None],
        },
        "DBSCAN": [
            {
                "reduce_dim": ["passthrough", PCA(n_components=PCA_COMPONENTS)],
                "cluster_algo": [DBSCAN()],
                "cluster_algo__eps": [eps],
                "cluster_algo__min_samples": [min_samples],
            }
            for eps, min_samples in zip(EPS_VALUES, MIN_SAMPLES_VALUES)
        ],
        "OPTICS": {
            "reduce_dim": ["passthrough", PCA(n_components=PCA_COMPONENTS)],
            "cluster_algo": [OPTICS(cluster_method="dbscan")],
            "cluster_algo__min_samples": MIN_SAMPLES_VALUES,
        },
        "GaussianMixture": {
            "reduce_dim": ["passthrough", PCA(n_components=PCA_COMPONENTS)],
            "reduce_dim": [PCA(n_components=PCA_COMPONENTS)],
            "cluster_algo": [GaussianMixture(random_state=SEED)],
            "cluster_algo__n_components": K_VALUES,
            "cluster_algo__covariance_type": COVARIANCE_TYPE_VALUES,
        },
        "SOM": [
            {
                "reduce_dim": ["passthrough", PCA(n_components=PCA_COMPONENTS)],
                "cluster_algo": [SOM(random_state=SEED)],
                "cluster_algo__n_columns": [k1],
                "cluster_algo__n_rows": [k2],
                "cluster_algo__initialcodebook": SOM_INITIALCODEBOOK_VALUES,
                "cluster_algo__neighborhood": SOM_NEIGHBORHOOD_VALUES,
            }
            for k1, k2 in product([1] + list(K_VALUES), list(K_VALUES))
            if k1 * k2 <= max(K_VALUES)
        ],
        "AffinityPropagation": {
            "reduce_dim": ["passthrough", PCA(n_components=PCA_COMPONENTS)],
            "cluster_algo": [AffinityPropagation(random_state=SEED)],
            "cluster_algo__damping": AFFINITY_PROP_DUMPING_VALUES,
            "cluster_algo__preference": AFFINITY_PROP_PREFERENCE_VALUES,
        },
        "SpectralClustering": [
            {
                "reduce_dim": ["passthrough", PCA(n_components=PCA_COMPONENTS)],
                "cluster_algo": [SpectralClustering(random_state=SEED)],
                "cluster_algo__n_clusters": K_VALUES,
                "cluster_algo__affinity": ["rbf"],
                "cluster_algo__assign_labels": ["kmeans", "discretize", "cluster_qr"],
            },
            {
                "reduce_dim": ["passthrough", PCA(n_components=PCA_COMPONENTS)],
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


def clusters_count_scorer(
    estimator: BaseEstimator,
    X: Union[pd.DataFrame, np.array],
    y: Union[pd.Series, np.array],
) -> int:
    labels = estimator.fit_predict(X)
    return len(np.unique(labels))
