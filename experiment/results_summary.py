from argparse import ArgumentParser
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
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
from sklearn.manifold import TSNE
from sklearn.mixture import GaussianMixture
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import QuantileTransformer, StandardScaler
from sklearn_extra.cluster import KMedoids
from sklearn_som.som import SOM
from tqdm.auto import tqdm
from umap.umap_ import UMAP
from utils import SEED, load_gemler_data_normed, load_metabric_data_normed

DATASETS = {
    "GEMLER": load_gemler_data_normed(QuantileTransformer()),
    "METABRIC": load_metabric_data_normed(QuantileTransformer()),
}

SCORES = [
    ("silhouette", lambda x: max(x[~x.isna()], default=None)),
    ("calinski_harabasz", lambda x: max(x[~x.isna()], default=None)),
    ("davies_bouldin", lambda x: min(x[~x.isna()], default=None)),
    ("adjusted_rand_index", lambda x: max(x[(x != 0) & (~x.isna())], default=None)),
    ("adjusted_mutual_info", lambda x: max(x[(x > 0) & (~x.isna())], default=None)),
]


def path_list_parse(arg: str, delim: str = ",") -> list[Path]:
    return [Path(a) for a in arg.split(delim)]


def boxplot_mean_scores_per_algo(results_df: pd.DataFrame, save_path: Path) -> None:
    n_cols = 3
    n_rows = int(len(SCORES) // n_cols + 1 * (len(SCORES) % n_cols > 0))
    fig = plt.figure(figsize=(n_rows * 10, n_cols * 5))
    for n, (score, _) in enumerate(SCORES):
        ax = fig.add_subplot(n_rows, n_cols, n + 1)
        sns.boxplot(
            data=results_df,
            x="algo_name",
            y=f"mean_test_{score}",
            hue="results_tag"
            if results_df.loc[:, "results_tag"].unique().shape[0] > 1
            else None,
            ax=ax,
        )
        ax.set_title(" ".join([s.capitalize() for s in score.split("_")]))
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
    fig.suptitle("Mean scores for differnt hyperparameters")
    plt.tight_layout()
    fig.savefig(save_path)


def boxplot_scores_per_best_config(results_df: pd.DataFrame, save_path: Path) -> None:
    n_cols = 3
    n_rows = int(len(SCORES) // n_cols + 1 * (len(SCORES) % n_cols > 0))
    fig = plt.figure(figsize=(n_rows * 10, n_cols * 5))
    for n, (score, transform_func) in enumerate(SCORES):
        max_ids = (
            results_df.groupby(["algo_name", "results_tag"])[
                f"mean_test_{score}"
            ].transform(
                transform_func,
            )
            == results_df.loc[:, f"mean_test_{score}"]
        )
        max_results_df = (
            results_df.loc[max_ids]
            .filter(regex=f"(split[0-9]+_test_{score}|algo_name|results_tag)")
            .melt(id_vars=["algo_name", "results_tag"])
        )

        ax = fig.add_subplot(n_rows, n_cols, n + 1)
        sns.boxplot(
            data=max_results_df,
            x="algo_name",
            y=f"value",
            hue="results_tag"
            if max_results_df.loc[:, "results_tag"].unique().shape[0] > 1
            else None,
            ax=ax,
        )
        ax.set_title(" ".join([s.capitalize() for s in score.split("_")]))
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
    fig.suptitle("Results for best hyperparameters for each score")
    plt.tight_layout()
    fig.savefig(save_path)


def get_best_labels_per_score_per_algo(
    results_df: pd.DataFrame,
    data_df: pd.DataFrame,
) -> dict[str, dict[str, pd.Series]]:
    labels_per_score_per_algo = {}
    for score, transform_func in tqdm(SCORES):
        labels_per_score_per_algo[score] = {}
        best_ids = (
            results_df.groupby("algo_name")[f"mean_test_{score}"].transform(
                transform_func,
            )
            == results_df.loc[:, f"mean_test_{score}"]
        )
        best_results_params = results_df.loc[best_ids, "params"].map(lambda x: eval(x))

        for param_dict in best_results_params:
            reduce_dim = (
                param_dict["reduce_dim"]
                if "reduce_dim" in param_dict
                else "passthrough"
            )
            cluster_algo = param_dict["cluster_algo"]
            for key, value in param_dict.items():
                if key in ["cluster_algo", "reduce_dim"]:
                    continue
                setattr(cluster_algo, key[len("cluster_algo__") :], value)
            pipeline = Pipeline(
                [("reduce_dim", reduce_dim), ("cluster_algo", cluster_algo)]
            )

            labels_per_score_per_algo[score][
                cluster_algo.__class__.__name__
            ] = pd.Series(pipeline.fit_predict(data_df), index=data_df.index).map(str)
    return labels_per_score_per_algo


def embedding_plot_per_best_config(
    labels_per_score_per_algo: dict[str, dict[str, pd.Series]],
    embedding: pd.DataFrame,
    ground_truth: Optional[pd.Series],
    save_dir: Path,
) -> None:
    for score, labels_per_algo in tqdm(labels_per_score_per_algo.items()):
        n_cols = 3
        n_plots = len(labels_per_algo) + 1 * (ground_truth is not None)
        n_rows = int(n_plots // n_cols + 1 * (n_plots % n_cols > 0))
        fig = plt.figure(figsize=(n_rows * 10, n_cols * 5))

        for n, (algo_name, labels) in enumerate(labels_per_algo.items()):
            ax = fig.add_subplot(n_rows, n_cols, n + 1)
            sns.scatterplot(
                data=embedding,
                x="x",
                y="y",
                hue=labels,
                ax=ax,
            )
            ax.set_title(algo_name)
        if ground_truth is not None:
            ax = fig.add_subplot(n_rows, n_cols, n + 2)
            sns.scatterplot(
                data=embedding,
                x="x",
                y="y",
                hue=ground_truth.map(str),
                ax=ax,
            )
            ax.set_title("Ground truth")
        fig.suptitle(f"Embedding visualization for best hyperparameters for {score}")
        plt.tight_layout()
        fig.savefig(save_dir / f"Embedding_best_hyperparameters_{score}.png")


def joined_results_dir(results_dirs: list[Path]):
    return results_dirs[0].parent / "_vs_".join([r.name for r in results_dirs])


def main(
    results_dirs: list[Path], datasets: list[str], skip_embedding: bool = False
) -> None:
    for dataset in datasets:
        results_dfs = []
        for i, results_dir in enumerate(results_dirs):
            results_dfs.append(
                pd.read_csv(
                    results_dir / f"{dataset}_grid_search.csv", index_col=0
                ).reset_index()
            )
            results_dfs[i]["results_tag"] = results_dir.name
        results_df = pd.concat(results_dfs)

        output_dir = joined_results_dir(results_dirs)
        output_dir.mkdir(exist_ok=True)

        boxplot_mean_scores_per_algo(
            results_df,
            output_dir / f"{dataset}_mean_score_per_algo.png",
        )

        boxplot_scores_per_best_config(
            results_df,
            output_dir / f"{dataset}_best_config_scores.png",
        )

        if not skip_embedding and len(results_dirs) == 1:
            data_df, ground_truth = DATASETS[dataset]()

            labels_per_score_per_algo = get_best_labels_per_score_per_algo(
                results_df,
                data_df,
            )

            embedding = pd.DataFrame(
                TSNE(
                    n_components=2, random_state=SEED, perplexity=30, metric="manhattan"
                ).fit_transform(data_df),
                index=data_df.index,
                columns=["x", "y"],
            )

            tsne_dir = results_dir / f"{dataset}_tsne"
            tsne_dir.mkdir(exist_ok=True)

            embedding_plot_per_best_config(
                labels_per_score_per_algo,
                embedding,
                ground_truth,
                tsne_dir,
            )

            embedding = pd.DataFrame(
                PCA(n_components=2, random_state=SEED).fit_transform(data_df),
                index=data_df.index,
                columns=["x", "y"],
            )

            pca_dir = results_dir / f"{dataset}_pca"
            pca_dir.mkdir(exist_ok=True)

            embedding_plot_per_best_config(
                labels_per_score_per_algo,
                embedding,
                ground_truth,
                pca_dir,
            )

            embedding = pd.DataFrame(
                UMAP(n_components=2, random_state=SEED).fit_transform(data_df),
                index=data_df.index,
                columns=["x", "y"],
            )

            umap_dir = results_dir / f"{dataset}_umap"
            umap_dir.mkdir(exist_ok=True)

            embedding_plot_per_best_config(
                labels_per_score_per_algo,
                embedding,
                ground_truth,
                umap_dir,
            )


if __name__ == "__main__":
    argparser = ArgumentParser()
    argparser.add_argument("results_dirs", type=path_list_parse)
    argparser.add_argument(
        "--datasets", type=lambda x: x.split(","), default=list(DATASETS.keys())
    )
    argparser.add_argument("--skip_embedding", default=False, action="store_true")
    kwargs = dict(argparser.parse_args()._get_kwargs())
    main(**kwargs)
