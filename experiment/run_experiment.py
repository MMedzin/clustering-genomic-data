from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import logging
from datetime import datetime
from pathlib import Path
from typing import Union

import pandas as pd
from clustering_grid_search import ClusteringGridSearchCV
from sklearn.metrics import (
    adjusted_mutual_info_score,
    adjusted_rand_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import BaseCrossValidator
from utils import (
    WholeDatasetCV,
    clusters_count_label_scorer,
    get_datasets_dict,
    make_clustering_label_scorer_supervised,
    make_clustering_label_scorer_unsupervised,
    SEED,
)

DEFAULT_EXPERIMENT_DIR = Path(__file__).parent

START_TIMESTAMP = datetime.now().strftime("%Y_%m_%d_%H%M")

logging.basicConfig(
    filename=f"Experiment_{START_TIMESTAMP}.log",
    filemode="a",
    format="%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
    level=logging.INFO,
)


DEFAULT_CV_REPEATS = 5
DEFAULT_CV_SCHEME = WholeDatasetCV(
    n_repeats=DEFAULT_CV_REPEATS, shuffle_each_repeat=True, random_state=SEED
)


SCORERS = {
    "silhouette": make_clustering_label_scorer_unsupervised(silhouette_score),
    "calinski_harabasz": make_clustering_label_scorer_unsupervised(
        calinski_harabasz_score
    ),
    "davies_bouldin": make_clustering_label_scorer_unsupervised(davies_bouldin_score),
    "adjusted_rand_index": make_clustering_label_scorer_supervised(adjusted_rand_score),
    "adjusted_mutual_info": make_clustering_label_scorer_supervised(
        adjusted_mutual_info_score
    ),
    "clusters_count": clusters_count_label_scorer,
}


def get_grid_search_kwargs(
    param_grid: Union[list[dict], dict], cv: BaseCrossValidator = DEFAULT_CV_SCHEME
) -> dict:
    return dict(
        estimator=Pipeline([("reduce_dim", "passthrough"), ("cluster_algo", None)]),
        param_grid=param_grid,
        cv=cv,
        scoring=SCORERS,
        n_jobs=-1,
        verbose=2,
        refit=list(SCORERS.keys())[0],
    )


def main(
    scaler: str = "min-max",
    results_dir_tag: str = "",
    experiment_dir: Path = DEFAULT_EXPERIMENT_DIR,
    repeats: int = DEFAULT_CV_REPEATS,
    shuffle_each_repeat: bool = True,
) -> None:
    logging.info("Starting experiment...")
    tag = f"_{results_dir_tag}" if results_dir_tag else ""
    results_dir = experiment_dir / f"results_{START_TIMESTAMP}{tag}"
    results_dir.mkdir(exist_ok=True)

    cv = WholeDatasetCV(
        n_repeats=repeats, shuffle_each_repeat=shuffle_each_repeat, random_state=SEED
    )

    datasets = get_datasets_dict(scaler, include_param_grid=True)
    for name, (data_loader, param_grid_loader) in datasets.items():
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
                    grid_search = ClusteringGridSearchCV(
                        **get_grid_search_kwargs(algo_grid, cv=cv)
                    )
                    grid_search.fit(
                        data.values, ground_truth_encoded.loc[data.index].values
                    )
                    results_df = pd.DataFrame(grid_search.cv_results_)
                    results_df = results_df.loc[:, results_df.columns.sort_values()]
                    results_df["algo_name"] = algo_name
                    results_df.loc[
                        :, ~results_df.columns.str.startswith("param_")
                    ].to_csv(
                        results_dir / f"{name}_grid_search.csv",
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
    argparser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    argparser.add_argument(
        "--scaler",
        default="min-max",
        choices=["min-max", "pca", "quantile", "standard"],
        help="select scaler used for data preprocessing",
    )
    argparser.add_argument(
        "--results_dir_tag",
        default="",
        type=str,
        help="tag added as suffix to the results directory name",
    )
    argparser.add_argument(
        "--experiment_dir",
        default=DEFAULT_EXPERIMENT_DIR,
        type=Path,
        help="experiment directory, inside of which results directory will be created",
    )
    argparser.add_argument(
        "--repeats",
        default=DEFAULT_CV_REPEATS,
        type=int,
        help="number of repeats of the evaluation",
    )
    argparser.add_argument(
        "--shuffle_each_repeat",
        default=False,
        action="store_true",
        help="flags if the order of samples in the dataset should be shuffled in each repeat",
    )
    kwargs = dict(argparser.parse_args()._get_kwargs())
    main(**kwargs)
