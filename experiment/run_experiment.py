import logging
from datetime import datetime
from pathlib import Path
from typing import Union

import pandas as pd
from sklearn.metrics import (
    adjusted_mutual_info_score,
    adjusted_rand_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
)
from sklearn.model_selection import GridSearchCV, RepeatedStratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, QuantileTransformer, StandardScaler
from utils import (
    SEED,
    clusters_count_scorer,
    load_gemler_data_normed,
    load_gemler_normed_param_grid,
    load_metabric_data_normed,
    load_metabric_normed_param_grid,
    make_clustering_scorer_supervised,
    make_clustering_scorer_unsupervised,
)

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
    (
        "GEMLER",
        load_gemler_data_normed(),
        load_gemler_normed_param_grid,
    ),
    (
        "METABRIC",
        load_metabric_data_normed(),
        load_metabric_normed_param_grid,
    ),
]
# DATASETS = [
#     (
#         "GEMLER_StandardScaler",
#         load_gemler_data_normed(StandardScaler()),
#         load_gemler_normed_param_grid,
#     ),
#     (
#         "METABRIC_StandardScaler",
#         load_metabric_data_normed(StandardScaler()),
#         load_metabric_normed_param_grid,
#     ),
#     (
#         "GEMLER_QuantileScaler",
#         load_gemler_data_normed(QuantileTransformer()),
#         load_gemler_normed_param_grid,
#     ),
#     (
#         "METABRIC_QuantileScaler",
#         load_metabric_data_normed(QuantileTransformer()),
#         load_metabric_normed_param_grid,
#     ),
# ]

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
    "clusters_count": clusters_count_scorer,
}


def get_grid_search_kwargs(param_grid: Union[list[dict], dict]) -> dict:
    return dict(
        estimator=Pipeline([("reduce_dim", "passthrough"), ("cluster_algo", None)]),
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
            for i, (algo_name, algo_grid) in enumerate(param_grid_loader().items()):
                try:
                    grid_search = GridSearchCV(**get_grid_search_kwargs(algo_grid))
                    grid_search.fit(
                        data.values, ground_truth_encoded.loc[data.index].values
                    )
                    results_df = pd.DataFrame(grid_search.cv_results_)
                    results_df["algo_name"] = algo_name
                    results_df.loc[
                        :, ~results_df.columns.str.startswith("param_")
                    ].to_csv(
                        RESULTS_DIR / f"{name}_grid_search.csv",
                        mode="a",
                        header=(i == 0),
                    )
                    logging.info("%s GridSearch done for %s.", name, algo_name)
                except Exception as error:
                    logging.critical(error, exc_info=True)
                    logging.info(
                        "%s GridSearch failed for %s. Skipping...", name, algo_name
                    )
                    continue
            logging.info("%s GridSearch results saved to csv.", name)

        except Exception as error:
            logging.critical(error, exc_info=True)
            raise error


if __name__ == "__main__":
    main()
