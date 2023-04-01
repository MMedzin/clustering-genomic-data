import pandas as pd
import numpy as np
from utils import (
    SEED,
    load_gemler_data_normed,
    load_gemler_normed_param_grid,
    load_metabric_data_normed,
    load_metabric_normed_param_grid,
    make_clustering_scorer_supervised,
    make_clustering_scorer_unsupervised,
)
from sklearn.model_selection import GridSearchCV, RepeatedStratifiedKFold
from sklearn.cluster import (
    KMeans,
    AgglomerativeClustering,
    Birch,
    DBSCAN,
    OPTICS,
    AffinityPropagation,
)
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
    adjusted_mutual_info_score,
    adjusted_rand_score,
)
from somlearn.som import SOM
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

DATASETS = [
    ("GEMLER", load_gemler_data_normed, load_gemler_normed_param_grid),
    ("METABRIC", load_metabric_data_normed, load_metabric_normed_param_grid),
]

CV_SPLITS = 2
CV_REPEATS = 5
CV_SCHEME = RepeatedStratifiedKFold(
    n_repeats=CV_REPEATS, n_splits=CV_SPLITS, random_state=SEED
)


SCORERS = {
    "silhouette": make_clustering_scorer_unsupervised(silhouette_score),
    "calinski_harabasz": make_clustering_scorer_unsupervised(calinski_harabasz_score),
    "davies_bouldin": make_clustering_scorer_unsupervised(davies_bouldin_score),
    "adjusted_rand_index": make_clustering_scorer_supervised(adjusted_rand_score),
    "adjusted_mutual_info": make_clustering_scorer_supervised(
        adjusted_mutual_info_score
    ),
}


def get_grid_search_kwargs(param_grid: list[dict]) -> dict:
    return dict(
        estimator=Pipeline([("cluster_algo", None)]),
        param_grid=param_grid,
        cv=CV_SCHEME,
        scoring=SCORERS,
        n_jobs=-1,
        verbose=2,
        refit=list(SCORERS.keys())[0],
    )


def main() -> None:
    logging.info("Starting experiment...")
    for name, data_loader, param_grid_loader in DATASETS:
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
            grid_search = GridSearchCV(**get_grid_search_kwargs(param_grid_loader()))
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
