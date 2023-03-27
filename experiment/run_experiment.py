import pandas as pd
from utils import (
    load_gemler_data,
    make_clustering_scorer_supervised,
    make_clustering_scorer_unsupervised,
)
from sklearn.model_selection import GridSearchCV, RepeatedStratifiedKFold
from sklearn.cluster import KMeans, AgglomerativeClustering, Birch, DBSCAN, OPTICS
from sklearn.mixture import GaussianMixture
from sklearn_extra.cluster import KMedoids
import logging
from datetime import datetime
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    silhouette_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    rand_score,
    mutual_info_score,
)
from pathlib import Path

EXPERIMENT_DIR = Path(__file__).parent

START_TIMESTAMP = datetime.now().strftime("%Y_%m_%d_%H%M")

logging.basicConfig(
    filename=f"Experiment_{START_TIMESTAMP}.log",
    filemode="a",
    format="%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
    level=logging.INFO,
)

RESULTS_DIR = EXPERIMENT_DIR / f"results_{START_TIMESTAMP}"
RESULTS_DIR.mkdir(exist_ok=True)

DATASETS = [("GEMLER", load_gemler_data)]

CV_SPLITS = 2
CV_REPEATS = 5
CV_SCHEME = RepeatedStratifiedKFold(n_repeats=CV_REPEATS, n_splits=CV_SPLITS)

K_VALUES = [2, 3, 4, 5, 6, 7, 8]
K_MEANS_INIT = ["k-means++", "random"]
K_MEDOIDS_INIT = ["k-medoids++", "random"]
EPS_VALUES = [0.1, 0.5, 1, 2, 5, 10]
MIN_SAMPLES_VALUES = [2, 5, 10, 20, 50]
N_CLUSTER_VALUES = [2, 3, 4, 5, 6, 7, 8]
LINKAGE_VALUES = ["ward", "complete", "average"]
BIRCH_THRESHOLD_VALUES = [0.1, 0.5, 1, 2, 5, 10]
BIRCH_BRANCHING_FACTOR_VALUES = [5, 10, 20, 50, 100]
XI_VALUES = [0.01, 0.05, 0.1, 0.5, 1, 2]
N_COMPONENTS_VALUES = [2, 3, 4, 5, 6, 7, 8]
COVARIANCE_TYPE_VALUES = ["full", "tied", "diag", "spherical"]

PARAM_GRID = [
    {
        "cluster_algo": [KMeans(n_init="auto")],
        "cluster_algo__n_clusters": K_VALUES,
        "cluster_algo__init": K_MEANS_INIT,
    },
    {
        "cluster_algo": [KMedoids()],
        "cluster_algo__n_clusters": K_VALUES,
        "cluster_algo__init": K_MEDOIDS_INIT,
    },
    {
        "cluster_algo": [DBSCAN()],
        "cluster_algo__eps": EPS_VALUES,
        "cluster_algo__min_samples": MIN_SAMPLES_VALUES,
    },
    {
        "cluster_algo": [AgglomerativeClustering()],
        "cluster_algo__n_clusters": N_CLUSTER_VALUES,
        "cluster_algo__linkage": LINKAGE_VALUES,
    },
    {
        "cluster_algo": [Birch()],
        "cluster_algo__threshold": BIRCH_THRESHOLD_VALUES,
        "cluster_algo__branching_factor": BIRCH_BRANCHING_FACTOR_VALUES,
    },
    {
        "cluster_algo": [OPTICS()],
        "cluster_algo__xi": XI_VALUES,
        "cluster_algo__min_samples": MIN_SAMPLES_VALUES,
    },
    {
        "cluster_algo": [GaussianMixture()],
        "cluster_algo__n_components": N_COMPONENTS_VALUES,
        "cluster_algo__covariance_type": COVARIANCE_TYPE_VALUES,
    },
]


SCORERS = {
    "silhouette": make_clustering_scorer_unsupervised(silhouette_score),
    "calinski_harabasz": make_clustering_scorer_unsupervised(calinski_harabasz_score),
    "davies_bouldin": make_clustering_scorer_unsupervised(davies_bouldin_score),
    "rand_index": make_clustering_scorer_supervised(rand_score),
    "mutual_info": make_clustering_scorer_supervised(mutual_info_score),
}

GRID_SEARCH_KWARGS = dict(
    estimator=Pipeline([("cluster_algo", None)]),
    param_grid=PARAM_GRID,
    cv=CV_SCHEME,
    scoring=SCORERS,
    n_jobs=-1,
    verbose=2,
    refit=list(SCORERS.keys())[0],
)


def main() -> None:
    logging.info("Starting experiment...")
    for name, data_loader in DATASETS:
        try:
            logging.info("Loading %s data...", name)
            data, ground_truth = data_loader()
            label_encoder = LabelEncoder().fit(ground_truth)
            ground_truth_encoded = pd.Series(
                label_encoder.transform(ground_truth),
                index=ground_truth.index,
                name=ground_truth.name,
            )
            logging.info("%s was loaded.", name)
            logging.info("Running GridSearch for %s...", name)
            grid_search = GridSearchCV(**GRID_SEARCH_KWARGS)
            grid_search.fit(data, ground_truth_encoded)
            logging.info("%s GridSearch done.", name)
            pd.DataFrame(grid_search.cv_results_).to_csv(
                RESULTS_DIR / f"{name}_grid_search.csv"
            )
            logging.info("%s GridSearch results saved to csv.", name)

        except Exception as error:
            logging.critical(error, exc_info=True)
            raise error


if __name__ == "__main__":
    main()
