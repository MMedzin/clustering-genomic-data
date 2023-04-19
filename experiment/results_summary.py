from argparse import ArgumentParser
from pathlib import Path
from typing import Optional
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from umap import UMAP
from sklearn.cluster import (
    DBSCAN,
    OPTICS,
    AffinityPropagation,
    AgglomerativeClustering,
    Birch,
    KMeans,
    SpectralClustering,
)
from sklearn_extra.cluster import KMedoids
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from utils import load_gemler_data_normed, load_metabric_data_normed, SEED
from tqdm.auto import tqdm

DATASETS = {
    "GEMLER": load_gemler_data_normed,
    "METABRIC": load_metabric_data_normed,
}
SCORES = [
    ("silhouette", max),
    ("calinski_harabasz", max),
    ("davies_bouldin", min),
    ("adjusted_rand_index", lambda x: max(x[x != 0], default=None)),
    ("adjusted_mutual_info", lambda x: max(x[x > 0], default=None)),
]


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
            results_df.groupby("algo_name")[f"mean_test_{score}"].transform(
                transform_func
            )
            == results_df.loc[:, f"mean_test_{score}"]
        )
        max_results_df = (
            results_df.loc[max_ids]
            .filter(regex=f"(split[0-9]+_test_{score}|algo_name)")
            .melt("algo_name")
        )

        ax = fig.add_subplot(n_rows, n_cols, n + 1)
        sns.boxplot(
            data=max_results_df,
            x="algo_name",
            y=f"value",
            ax=ax,
        )
        ax.set_title(" ".join([s.capitalize() for s in score.split("_")]))
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
    fig.suptitle("Results for best hyperparameters for each score")
    plt.tight_layout()
    fig.savefig(save_path)


def embedding_plot_per_best_config(
    results_df: pd.DataFrame,
    data_df: pd.DataFrame,
    embedding: pd.DataFrame,
    ground_truth: Optional[pd.Series],
    save_dir: Path,
) -> None:
    for score, transform_func in tqdm(SCORES):
        best_ids = (
            results_df.groupby("algo_name")[f"mean_test_{score}"].transform(
                transform_func
            )
            == results_df.loc[:, f"mean_test_{score}"]
        )
        best_results_params = results_df.loc[best_ids, "params"].map(lambda x: eval(x))
        n_cols = 3
        n_plots = len(best_results_params) + 1 * (ground_truth is not None)
        n_rows = int(n_plots // n_cols + 1 * (n_plots % n_cols > 0))
        fig = plt.figure(figsize=(n_rows * 10, n_cols * 5))

        for n, param_dict in enumerate(best_results_params):
            cluster_algo = param_dict["cluster_algo"]
            for key, value in param_dict.items():
                if key == "cluster_algo":
                    continue
                setattr(cluster_algo, key[len("cluster_algo__") :], value)

            labels = pd.Series(
                cluster_algo.fit_predict(data_df), index=data_df.index
            ).map(str)

            ax = fig.add_subplot(n_rows, n_cols, n + 1)
            sns.scatterplot(
                data=embedding,
                x="x",
                y="y",
                hue=labels,
                ax=ax,
            )
            ax.set_title(str(cluster_algo))
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


def main(results_dir: Path, datasets: list[str]) -> None:
    for dataset in datasets:
        results_df = pd.read_csv(
            results_dir / f"{dataset}_grid_search.csv", index_col=0
        ).reset_index()

        boxplot_mean_scores_per_algo(
            results_df, results_dir / f"{dataset}_mean_score_per_algo.png"
        )

        boxplot_scores_per_best_config(
            results_df,
            results_dir / f"{dataset}_best_config_scores.png",
        )

        data_df, ground_truth = DATASETS[dataset]()

        embedding = pd.DataFrame(
            TSNE(n_components=2, random_state=SEED).fit_transform(data_df),
            index=data_df.index,
            columns=["x", "y"],
        )

        tsne_dir = results_dir / f"{dataset}_tsne"
        tsne_dir.mkdir(exist_ok=True)

        embedding_plot_per_best_config(
            results_df,
            data_df,
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
            results_df,
            data_df,
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
            results_df,
            data_df,
            embedding,
            ground_truth,
            umap_dir,
        )


if __name__ == "__main__":
    argparser = ArgumentParser()
    argparser.add_argument("results_dir", type=Path)
    argparser.add_argument(
        "--datasets", type=lambda x: x.split(","), default=list(DATASETS.keys())
    )
    kwargs = dict(argparser.parse_args()._get_kwargs())
    main(**kwargs)
