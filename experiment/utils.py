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
from sklearn.model_selection import BaseCrossValidator
from sklearn.neighbors import LocalOutlierFactor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, MinMaxScaler
from sklearn_extra.cluster import KMedoids
from sklearn_som.som import SOM

# from somlearn.som import SOM

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


def build_param_grid(
    reduce_dim: list[str, TransformerMixin],
    k_values: np.array,
    k_means_init: list[str],
    k_medoids_init: list[str],
    linkage_values: list[str],
    birch_threshold_values: list[float],
    birch_branching_factor_values: list[int],
    eps_values: list[float],
    min_samples_values: list[int],
    covariance_type_values: list[str],
    som_epochs: list[int],
    affinity_prop_dumping_values: np.array,
    affinity_prop_preference_values: list[float],
    pca_components: int,
    features_count: int,
) -> dict:
    return {
        "KMeans": {
            "reduce_dim": reduce_dim,
            "cluster_algo": [KMeans(n_init="auto", random_state=SEED)],
            "cluster_algo__n_clusters": k_values,
            "cluster_algo__init": k_means_init,
        },
        "KMedoids": {
            "reduce_dim": reduce_dim,
            "cluster_algo": [KMedoids(random_state=SEED)],
            "cluster_algo__n_clusters": k_values,
            "cluster_algo__init": k_medoids_init,
        },
        "AgglomerativeClustering": {
            "reduce_dim": reduce_dim,
            "cluster_algo": [AgglomerativeClustering()],
            "cluster_algo__n_clusters": k_values,
            "cluster_algo__linkage": linkage_values,
        },
        "Birch": {
            "reduce_dim": reduce_dim,
            "cluster_algo": [Birch()],
            "cluster_algo__threshold": birch_threshold_values,
            "cluster_algo__branching_factor": birch_branching_factor_values,
            "cluster_algo__n_clusters": list(k_values) + [None],
        },
        "DBSCAN": [
            {
                "reduce_dim": reduce_dim,
                "cluster_algo": [DBSCAN()],
                "cluster_algo__eps": [eps],
                "cluster_algo__min_samples": [min_samples],
            }
            for eps, min_samples in zip(eps_values, min_samples_values)
        ],
        "OPTICS": {
            "reduce_dim": reduce_dim,
            "cluster_algo": [OPTICS(cluster_method="xi", metric="euclidean")],
            "cluster_algo__min_samples": min_samples_values,
        },
        "GaussianMixture": {
            "reduce_dim": [PCA(n_components=pca_components)],
            "cluster_algo": [GaussianMixture(random_state=SEED)],
            "cluster_algo__n_components": k_values,
            "cluster_algo__covariance_type": covariance_type_values,
        },
        "SOM": [  # sklearn_som version
            {
                "reduce_dim": reduce_dim,
                "cluster_algo": [SOM(dim=features_count, random_state=SEED)],
                "cluster_algo__epochs": som_epochs,
                "cluster_algo__m": [k1],
                "cluster_algo__n": [k2],
            }
            for k1, k2 in product([1] + list(k_values), list(k_values))
            if k1 * k2 <= max(k_values)
        ],
        "AffinityPropagation": {
            "reduce_dim": reduce_dim,
            "cluster_algo": [AffinityPropagation(random_state=SEED)],
            "cluster_algo__damping": affinity_prop_dumping_values,
            "cluster_algo__preference": affinity_prop_preference_values,
        },
        "SpectralClustering": [
            {
                "reduce_dim": reduce_dim,
                "cluster_algo": [SpectralClustering(random_state=SEED)],
                "cluster_algo__n_clusters": k_values,
                "cluster_algo__affinity": ["rbf"],
                "cluster_algo__assign_labels": ["kmeans", "discretize", "cluster_qr"],
            },
            {
                "reduce_dim": reduce_dim,
                "cluster_algo": [SpectralClustering(random_state=SEED)],
                "cluster_algo__n_clusters": k_values,
                "cluster_algo__affinity": ["nearest_neighbors"],
                "cluster_algo__n_neighbors": min_samples_values,
                "cluster_algo__assign_labels": ["kmeans", "discretize", "cluster_qr"],
            },
        ],
    }


def load_gemler_minmax_param_grid() -> list[dict]:
    """
    Param grid created with help of jupyter notebook: hyperparameters_exploration_gmler.ipynb
    """

    K_VALUES = np.arange(2, 14, 1)
    K_MEANS_INIT = ["k-means++", "random"]
    K_MEDOIDS_INIT = ["k-medoids++", "random"]
    LINKAGE_VALUES = ["ward", "complete", "average", "single"]
    BIRCH_THRESHOLD_VALUES = [
        0.2642900288766189,
        0.5285800577532378,
        0.7928700866298567,
        1.0571601155064756,
        1.3214501443830946,
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
    SOM_EPOCHS = [10, 50, 100, 200, 300]
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
    FEATURES_COUNT = 7888

    return build_param_grid(
        reduce_dim=["passthrough"],
        k_values=K_VALUES,
        k_means_init=K_MEANS_INIT,
        k_medoids_init=K_MEDOIDS_INIT,
        linkage_values=LINKAGE_VALUES,
        birch_threshold_values=BIRCH_THRESHOLD_VALUES,
        birch_branching_factor_values=BIRCH_BRANCHING_FACTOR_VALUES,
        eps_values=EPS_VALUES,
        min_samples_values=MIN_SAMPLES_VALUES,
        covariance_type_values=COVARIANCE_TYPE_VALUES,
        som_epochs=SOM_EPOCHS,
        affinity_prop_dumping_values=AFFINITY_PROP_DUMPING_VALUES,
        affinity_prop_preference_values=AFFINITY_PROP_PREFERENCE_VALUES,
        pca_components=PCA_COMPONENTS,
        features_count=FEATURES_COUNT,
    )


def load_gemler_standard_param_grid() -> list[dict]:
    """
    Param grid created with help of jupyter notebook: hyperparameters_exploration_gmler.ipynb
    """

    K_VALUES = np.arange(2, 11, 1)
    K_MEANS_INIT = ["k-means++", "random"]
    K_MEDOIDS_INIT = ["k-medoids++", "random"]
    LINKAGE_VALUES = ["ward", "complete", "average", "single"]
    BIRCH_THRESHOLD_VALUES = [
        2.86387821312995,
        5.7277564262599,
        8.591634639389849,
        11.4555128525198,
        14.31939106564975,
    ]
    BIRCH_BRANCHING_FACTOR_VALUES = [5, 28, 52, 76, 100]
    EPS_VALUES = [
        106.34092085433089,
        115.31664757312046,
        117.58484396768434,
        118.50768307048276,
        119.61565367510651,
    ]
    MIN_SAMPLES_VALUES = [3, 11, 20, 28, 37]
    COVARIANCE_TYPE_VALUES = ["full", "tied", "diag", "spherical"]
    SOM_EPOCHS = [10, 50, 100, 200, 300]
    AFFINITY_PROP_DUMPING_VALUES = np.arange(0.5, 1, 0.1)
    AFFINITY_PROP_PREFERENCE_VALUES = [
        -29808.75520987832,
        -26496.671297669618,
        -23184.587385460916,
        -19872.50347325221,
        -16560.419561043513,
        -13248.335648834807,
        -9936.251736626105,
        -6624.167824417404,
        -3312.083912208702,
        0.0,
    ]
    PCA_COMPONENTS = 217
    FEATURES_COUNT = 7888

    return build_param_grid(
        reduce_dim=["passthrough"],
        k_values=K_VALUES,
        k_means_init=K_MEANS_INIT,
        k_medoids_init=K_MEDOIDS_INIT,
        linkage_values=LINKAGE_VALUES,
        birch_threshold_values=BIRCH_THRESHOLD_VALUES,
        birch_branching_factor_values=BIRCH_BRANCHING_FACTOR_VALUES,
        eps_values=EPS_VALUES,
        min_samples_values=MIN_SAMPLES_VALUES,
        covariance_type_values=COVARIANCE_TYPE_VALUES,
        som_epochs=SOM_EPOCHS,
        affinity_prop_dumping_values=AFFINITY_PROP_DUMPING_VALUES,
        affinity_prop_preference_values=AFFINITY_PROP_PREFERENCE_VALUES,
        pca_components=PCA_COMPONENTS,
        features_count=FEATURES_COUNT,
    )


def load_gemler_quantile_param_grid() -> list[dict]:
    """
    Param grid created with help of jupyter notebook: hyperparameters_exploration_gmler.ipynb
    """

    K_VALUES = np.arange(2, 13, 1)
    K_MEANS_INIT = ["k-means++", "random"]
    K_MEDOIDS_INIT = ["k-medoids++", "random"]
    LINKAGE_VALUES = ["ward", "complete", "average", "single"]
    BIRCH_THRESHOLD_VALUES = [
        0.5622221243895874,
        1.1244442487791748,
        1.6866663731687623,
        2.2488884975583496,
        2.811110621947937,
    ]
    BIRCH_BRANCHING_FACTOR_VALUES = [5, 28, 52, 76, 100]
    EPS_VALUES = [
        17.66276436582734,
        30.43073941695291,
        31.544987670431624,
        31.955367749298368,
        32.37894388525505,
    ]
    MIN_SAMPLES_VALUES = [3, 11, 20, 28, 37]
    COVARIANCE_TYPE_VALUES = ["full", "tied", "diag", "spherical"]
    SOM_EPOCHS = [10, 50, 100, 200, 300]
    AFFINITY_PROP_DUMPING_VALUES = np.arange(0.5, 1, 0.1)
    AFFINITY_PROP_PREFERENCE_VALUES = [
        -2621.5701525461864,
        -2330.2845800410546,
        -2038.9990075359228,
        -1747.7134350307908,
        -1456.427862525659,
        -1165.1422900205273,
        -873.8567175153953,
        -582.5711450102635,
        -291.28557250513177,
        0.0,
    ]
    PCA_COMPONENTS = 169
    FEATURES_COUNT = 7888

    return build_param_grid(
        reduce_dim=["passthrough"],
        k_values=K_VALUES,
        k_means_init=K_MEANS_INIT,
        k_medoids_init=K_MEDOIDS_INIT,
        linkage_values=LINKAGE_VALUES,
        birch_threshold_values=BIRCH_THRESHOLD_VALUES,
        birch_branching_factor_values=BIRCH_BRANCHING_FACTOR_VALUES,
        eps_values=EPS_VALUES,
        min_samples_values=MIN_SAMPLES_VALUES,
        covariance_type_values=COVARIANCE_TYPE_VALUES,
        som_epochs=SOM_EPOCHS,
        affinity_prop_dumping_values=AFFINITY_PROP_DUMPING_VALUES,
        affinity_prop_preference_values=AFFINITY_PROP_PREFERENCE_VALUES,
        pca_components=PCA_COMPONENTS,
        features_count=FEATURES_COUNT,
    )


def load_gemler_pca_param_grid() -> list[dict]:
    """
    Param grid created with help of jupyter notebook: hyperparameters_exploration_gmler.ipynb
    """

    K_VALUES = np.arange(2, 12, 1)
    K_MEANS_INIT = ["k-means++", "random"]
    K_MEDOIDS_INIT = ["k-medoids++", "random"]
    LINKAGE_VALUES = ["ward", "complete", "average", "single"]
    BIRCH_THRESHOLD_VALUES = [
        2.8066787626099616,
        5.613357525219923,
        8.420036287829884,
        11.226715050439847,
        14.033393813049807,
    ]
    BIRCH_BRANCHING_FACTOR_VALUES = [5, 28, 52, 76, 100]
    EPS_VALUES = [
        83.36569396731637,
        88.1786507390501,
        95.09903517071724,
        96.41025080199863,
        101.2507266061472,
    ]
    MIN_SAMPLES_VALUES = [3, 11, 20, 28, 37]
    COVARIANCE_TYPE_VALUES = ["full", "tied", "diag", "spherical"]
    SOM_EPOCHS = [10, 50, 100, 200, 300]
    AFFINITY_PROP_DUMPING_VALUES = np.arange(0.5, 1, 0.1)
    AFFINITY_PROP_PREFERENCE_VALUES = [
        -22997.734023722987,
        -20442.43024330932,
        -17887.126462895656,
        -15331.822682481992,
        -12776.518902068326,
        -10221.21512165466,
        -7665.911341240997,
        -5110.607560827331,
        -2555.3037804136657,
        0.0,
    ]
    PCA_COMPONENTS = 217
    FEATURES_COUNT = 7888

    return build_param_grid(
        reduce_dim=[PCA(n_components=PCA_COMPONENTS)],
        k_values=K_VALUES,
        k_means_init=K_MEANS_INIT,
        k_medoids_init=K_MEDOIDS_INIT,
        linkage_values=LINKAGE_VALUES,
        birch_threshold_values=BIRCH_THRESHOLD_VALUES,
        birch_branching_factor_values=BIRCH_BRANCHING_FACTOR_VALUES,
        eps_values=EPS_VALUES,
        min_samples_values=MIN_SAMPLES_VALUES,
        covariance_type_values=COVARIANCE_TYPE_VALUES,
        som_epochs=SOM_EPOCHS,
        affinity_prop_dumping_values=AFFINITY_PROP_DUMPING_VALUES,
        affinity_prop_preference_values=AFFINITY_PROP_PREFERENCE_VALUES,
        pca_components=PCA_COMPONENTS,
        features_count=FEATURES_COUNT,
    )


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


def load_metabric_minmax_param_grid() -> list[dict]:
    """
    Param grid created with help of jupyter notebook: hyperparameters_exploration_metabric.ipynb
    """

    K_VALUES = np.arange(2, 19, 1)
    K_MEANS_INIT = ["k-means++", "random"]
    K_MEDOIDS_INIT = ["k-medoids++", "random"]
    LINKAGE_VALUES = ["ward", "complete", "average", "single"]
    BIRCH_THRESHOLD_VALUES = [
        0.45665479428002376,
        0.9133095885600475,
        1.3699643828400712,
        1.826619177120095,
        2.283273971400119,
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
    SOM_EPOCHS = [10, 50, 100, 200, 300]
    AFFINITY_PROP_DUMPING_VALUES = np.arange(0.5, 1, 0.1)

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
    FEATURES_COUNT = 20000

    return build_param_grid(
        reduce_dim=["passthrough"],
        k_values=K_VALUES,
        k_means_init=K_MEANS_INIT,
        k_medoids_init=K_MEDOIDS_INIT,
        linkage_values=LINKAGE_VALUES,
        birch_threshold_values=BIRCH_THRESHOLD_VALUES,
        birch_branching_factor_values=BIRCH_BRANCHING_FACTOR_VALUES,
        eps_values=EPS_VALUES,
        min_samples_values=MIN_SAMPLES_VALUES,
        covariance_type_values=COVARIANCE_TYPE_VALUES,
        som_epochs=SOM_EPOCHS,
        affinity_prop_dumping_values=AFFINITY_PROP_DUMPING_VALUES,
        affinity_prop_preference_values=AFFINITY_PROP_PREFERENCE_VALUES,
        pca_components=PCA_COMPONENTS,
        features_count=FEATURES_COUNT,
    )


def load_metabric_standard_param_grid() -> list[dict]:
    """
    Param grid created with help of jupyter notebook: hyperparameters_exploration_metabric.ipynb
    """

    K_VALUES = np.arange(2, 15, 1)
    K_MEANS_INIT = ["k-means++", "random"]
    K_MEDOIDS_INIT = ["k-medoids++", "random"]
    LINKAGE_VALUES = ["ward", "complete", "average", "single"]
    BIRCH_THRESHOLD_VALUES = [
        3.8266374470120055,
        7.653274894024011,
        11.479912341036016,
        15.306549788048022,
        19.133187235060028,
    ]
    BIRCH_BRANCHING_FACTOR_VALUES = [5, 28, 52, 76, 100]
    EPS_VALUES = [
        171.5911457792247,
        182.7911020081354,
        184.74862793103273,
        185.33601032910138,
        183.04441137838256,
    ]
    MIN_SAMPLES_VALUES = [5, 16, 28, 40, 52]
    COVARIANCE_TYPE_VALUES = ["full", "tied", "diag", "spherical"]
    SOM_EPOCHS = [10, 50, 100, 200, 300]
    AFFINITY_PROP_DUMPING_VALUES = np.arange(0.5, 1, 0.1)
    AFFINITY_PROP_PREFERENCE_VALUES = [
        -76248.282402655,
        -67776.25102458222,
        -59304.21964650944,
        -50832.18826843666,
        -42360.15689036388,
        -33888.12551229111,
        -25416.09413421833,
        -16944.062756145548,
        -8472.031378072774,
        0.0,
    ]
    PCA_COMPONENTS = 273
    FEATURES_COUNT = 20000

    return build_param_grid(
        reduce_dim=["passthrough"],
        k_values=K_VALUES,
        k_means_init=K_MEANS_INIT,
        k_medoids_init=K_MEDOIDS_INIT,
        linkage_values=LINKAGE_VALUES,
        birch_threshold_values=BIRCH_THRESHOLD_VALUES,
        birch_branching_factor_values=BIRCH_BRANCHING_FACTOR_VALUES,
        eps_values=EPS_VALUES,
        min_samples_values=MIN_SAMPLES_VALUES,
        covariance_type_values=COVARIANCE_TYPE_VALUES,
        som_epochs=SOM_EPOCHS,
        affinity_prop_dumping_values=AFFINITY_PROP_DUMPING_VALUES,
        affinity_prop_preference_values=AFFINITY_PROP_PREFERENCE_VALUES,
        pca_components=PCA_COMPONENTS,
        features_count=FEATURES_COUNT,
    )


def load_metabric_quantile_param_grid() -> list[dict]:
    """
    Param grid created with help of jupyter notebook: hyperparameters_exploration_metabric.ipynb
    """

    K_VALUES = np.arange(2, 20, 1)
    K_MEANS_INIT = ["k-means++", "random"]
    K_MEDOIDS_INIT = ["k-medoids++", "random"]
    LINKAGE_VALUES = ["ward", "complete", "average", "single"]
    BIRCH_THRESHOLD_VALUES = [
        0.8243545803354415,
        1.648709160670883,
        2.4730637410063245,
        3.297418321341766,
        4.121772901677208,
    ]
    BIRCH_BRANCHING_FACTOR_VALUES = [5, 28, 52, 76, 100]
    EPS_VALUES = [
        43.19885467743711,
        44.89506786591525,
        52.91637193756566,
        53.22089431377895,
        53.63440331145333,
    ]
    MIN_SAMPLES_VALUES = [5, 16, 28, 40, 52]
    COVARIANCE_TYPE_VALUES = ["full", "tied", "diag", "spherical"]
    SOM_EPOCHS = [10, 50, 100, 200, 300]
    AFFINITY_PROP_DUMPING_VALUES = np.arange(0.5, 1, 0.1)
    AFFINITY_PROP_PREFERENCE_VALUES = [
        -6603.337474366642,
        -5869.633310548126,
        -5135.92914672961,
        -4402.224982911094,
        -3668.5208190925787,
        -2934.816655274063,
        -2201.1124914555467,
        -1467.408327637031,
        -733.7041638185156,
        0.0,
    ]
    PCA_COMPONENTS = 242
    FEATURES_COUNT = 20000

    return build_param_grid(
        reduce_dim=["passthrough"],
        k_values=K_VALUES,
        k_means_init=K_MEANS_INIT,
        k_medoids_init=K_MEDOIDS_INIT,
        linkage_values=LINKAGE_VALUES,
        birch_threshold_values=BIRCH_THRESHOLD_VALUES,
        birch_branching_factor_values=BIRCH_BRANCHING_FACTOR_VALUES,
        eps_values=EPS_VALUES,
        min_samples_values=MIN_SAMPLES_VALUES,
        covariance_type_values=COVARIANCE_TYPE_VALUES,
        som_epochs=SOM_EPOCHS,
        affinity_prop_dumping_values=AFFINITY_PROP_DUMPING_VALUES,
        affinity_prop_preference_values=AFFINITY_PROP_PREFERENCE_VALUES,
        pca_components=PCA_COMPONENTS,
        features_count=FEATURES_COUNT,
    )


def load_metabric_pca_param_grid() -> list[dict]:
    """
    Param grid created with help of jupyter notebook: hyperparameters_exploration_gmler.ipynb
    """

    K_VALUES = np.arange(2, 15, 1)
    K_MEANS_INIT = ["k-means++", "random"]
    K_MEDOIDS_INIT = ["k-medoids++", "random"]
    LINKAGE_VALUES = ["ward", "complete", "average", "single"]
    BIRCH_THRESHOLD_VALUES = [
        3.6398912746529795,
        7.279782549305959,
        10.919673823958938,
        14.559565098611918,
        18.199456373264898,
    ]
    BIRCH_BRANCHING_FACTOR_VALUES = [5, 28, 52, 76, 100]
    EPS_VALUES = [
        125.0869563929444,
        133.09023705675693,
        136.62484799312057,
        139.72696228137772,
        141.91261469172105,
        153.31780094478617,
        160.43398658659888,
    ]
    MIN_SAMPLES_VALUES = [5, 16, 28, 40, 52, 273, 546]
    COVARIANCE_TYPE_VALUES = ["full", "tied", "diag", "spherical"]
    SOM_EPOCHS = [10, 50, 100, 200, 300]
    AFFINITY_PROP_DUMPING_VALUES = np.arange(0.5, 1, 0.1)
    AFFINITY_PROP_PREFERENCE_VALUES = [
        -47120.001807035485,
        -41884.44605069821,
        -36648.890294360936,
        -31413.334538023657,
        -26177.778781686382,
        -20942.223025349107,
        -15706.667269011828,
        -10471.111512674557,
        -5235.555756337279,
        0.0,
    ]
    PCA_COMPONENTS = 273
    FEATURES_COUNT = 20000

    return build_param_grid(
        reduce_dim=[PCA(n_components=PCA_COMPONENTS)],
        k_values=K_VALUES,
        k_means_init=K_MEANS_INIT,
        k_medoids_init=K_MEDOIDS_INIT,
        linkage_values=LINKAGE_VALUES,
        birch_threshold_values=BIRCH_THRESHOLD_VALUES,
        birch_branching_factor_values=BIRCH_BRANCHING_FACTOR_VALUES,
        eps_values=EPS_VALUES,
        min_samples_values=MIN_SAMPLES_VALUES,
        covariance_type_values=COVARIANCE_TYPE_VALUES,
        som_epochs=SOM_EPOCHS,
        affinity_prop_dumping_values=AFFINITY_PROP_DUMPING_VALUES,
        affinity_prop_preference_values=AFFINITY_PROP_PREFERENCE_VALUES,
        pca_components=PCA_COMPONENTS,
        features_count=FEATURES_COUNT,
    )


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


def make_clustering_label_scorer_unsupervised(score_func: Callable) -> Callable:
    def scorer(labels: pd.Series, X: Union[pd.DataFrame, np.array], _y: Any) -> float:
        return score_func(X, labels)

    return scorer


def make_clustering_label_scorer_supervised(score_func: Callable) -> Callable:
    def scorer(
        labels: pd.Series,
        X: Union[pd.DataFrame, np.array],
        y: Union[pd.Series, np.array],
    ) -> float:
        return score_func(y, labels)

    return scorer


def clusters_count_scorer(
    estimator: BaseEstimator,
    X: Union[pd.DataFrame, np.array],
    y: Union[pd.Series, np.array],
) -> int:
    labels = estimator.fit_predict(X)
    return len(np.unique(labels))


def clusters_count_label_scorer(
    labels: pd.Series,
    X: Union[pd.DataFrame, np.array],
    y: Union[pd.Series, np.array],
) -> int:
    return len(np.unique(labels))


class WholeDatasetCV(BaseCrossValidator):
    def __init__(self, n_repeats=1, shuffle_each_repeat=False, random_state=None):
        self.n_repeats = n_repeats
        self.random_state = random_state
        self.shuffle_each_repeat = shuffle_each_repeat

    def get_n_splits(self, X=None, y=None, groups=None):
        return self.n_repeats

    def split(self, X, y=None, groups=None):
        n_samples = len(X)
        indices = np.arange(n_samples)
        rng = np.random.RandomState(self.random_state)
        rng.shuffle(indices)

        for _ in range(self.n_repeats):
            if self.shuffle_each_repeat:
                rng.shuffle(indices)
            yield indices, indices

    def __repr__(self):
        return f"{self.__class__.__name__}(n_repeats={self.n_repeats}, shuffle_each_repeat={self.shuffle_each_repeat}, random_state={self.random_state})"


def get_pipeline_from_params(param_dict: dict) -> Pipeline:
    reduce_dim = (
        param_dict["reduce_dim"] if "reduce_dim" in param_dict else "passthrough"
    )
    cluster_algo = param_dict["cluster_algo"]
    for key, value in param_dict.items():
        if key in ["cluster_algo", "reduce_dim"]:
            continue
        setattr(cluster_algo, key[len("cluster_algo__") :], value)
    return Pipeline([("reduce_dim", reduce_dim), ("cluster_algo", cluster_algo)])
