from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from pathlib import Path
from typing import Callable, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.legend import _get_legend_handles_labels
from matplotlib.patches import PathPatch
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
from sklearn.preprocessing import MinMaxScaler, QuantileTransformer, StandardScaler
from sklearn_extra.cluster import KMedoids
from sklearn_som.som import SOM
from tqdm.auto import tqdm
from umap.umap_ import UMAP
from utils import (
    SEED,
    get_datasets_dict,
    get_pipeline_from_params,
)

CLUSTERS_COUNT_STR = "clusters_count"

SCORES = {
    "silhouette": lambda x: max(x[~x.isna()], default=None),
    "calinski_harabasz": lambda x: max(x[~x.isna()], default=None),
    "davies_bouldin": lambda x: min(x[~x.isna()], default=None),
    "adjusted_rand_index": lambda x: max(x[(x != 0) & (~x.isna())], default=None),
    "adjusted_mutual_info": lambda x: max(x[(x > 0) & (~x.isna())], default=None),
    "adjusted_mutual_info": lambda x: max(x[(x > 0) & (~x.isna())], default=None),
}

SCORES_GROUPS = {
    "none": {"": (" ", list(SCORES.keys()))},
    "ground-truth": {
        "internal": (
            " internal ",
            ["silhouette", "calinski_harabasz", "davies_bouldin"],
        ),
        "external": (" external ", ["adjusted_rand_index", "adjusted_mutual_info"]),
    },
    "individual": {
        "silhouette": (None, ["silhouette"]),
        "calinski_harabasz": (None, ["calinski_harabasz"]),
        "davies_bouldin": (None, ["davies_bouldin"]),
        "adjusted_rand_index": (None, ["adjusted_rand_index"]),
        "adjusted_mutual_info": (None, ["adjusted_mutual_info"]),
    },
}

SCORES_PRETTY_NAMES = {
    "silhouette": "Silhouette Coeff.",
    "calinski_harabasz": "Calinski-Harabasz Index",
    "davies_bouldin": "Davies-Bouldin Index",
    "adjusted_rand_index": "Adjusted Rand Index",
    "adjusted_mutual_info": "Adjusted Mutual Info.",
    CLUSTERS_COUNT_STR: "clusters count",
}

ALGO_PRETTY_NAMES = {
    "KMeans": "K-Means",
    "KMedoids": "K-Medoids",
    "AffinityPropagation": "Affinity Propagation",
    "AgglomerativeClustering": "AHC",
    "Birch": "BIRCH",
    "GaussianMixture": "Gaussian Mixture",
    "DBSCAN": "DBSCAN",
    "OPTICS": "OPTICS",
    "SOM": "SOM",
    "SpectralClustering": "Spectral Clustering",
}

ALGO_GROUPS = {
    "K-Means": "Partition-based",
    "K-Medoids": "Partition-based",
    "Affinity Propagation": "Partition-based",
    "AHC": "Hierarchical",
    "BIRCH": "Hierarchical",
    "Gaussian Mixture": "Distribution-based",
    "DBSCAN": "Density-based",
    "OPTICS": "Density-based",
    "SOM": "Model-based",
    "Spectral Clustering": "Graph-based",
}

ALGO_ORDER = [
    "K-Means",
    "K-Medoids",
    "Affinity Propagation",
    "AHC",
    "BIRCH",
    "Gaussian Mixture",
    "DBSCAN",
    "OPTICS",
    "SOM",
    "Spectral Clustering",
]

ALGO_GROUPS_ORDER = [
    "Partition-based",
    "Hierarchical",
    "Distribution-based",
    "Density-based",
    "Model-based",
    "Graph-based",
]

PALETTE = sns.color_palette()

RESULTS_NAMES_DICT = {
    "results_2023_05_23_1732_PCA_WHOLE_DATA": "PCA",
    "results_2023_05_24_1020_StandardScaler_WHOLE_DATA": "StandardScaler",
    "results_2023_05_24_1508_QuantileTransformer_WHOLE_DATA": "QuantileTransformer",
    "results_2023_05_24_2340_MinMaxScaler_WHOLE_DATA": "MinMaxScaler",
}

UMAP_EMBEDDING_FILE = {
    "GEMLER": Path("./GEMLER_umap_embedding.csv"),
    "METABRIC": Path("./METABRIC_umap_embedding.csv"),
}
TSNE_EMBEDDING_FILE = {
    "GEMLER": Path("./GEMLER_tsne_embedding.csv"),
    "METABRIC": Path("./METABRIC_tsne_embedding.csv"),
}
PCA_EMBEDDING_FILE = {
    "GEMLER": Path("./GEMLER_pca_embedding.csv"),
    "METABRIC": Path("./METABRIC_pca_embedding.csv"),
}


def path_list_parse(arg: str, delim: str = ",") -> list[Path]:
    return [Path(a) for a in arg.split(delim)]


def boxplot_mean_scores_per_algo(
    results_df: pd.DataFrame,
    save_path: Path,
    score_groups_str: str = "none",
    additional_score_groups: Optional[list[tuple[str, tuple[str, list[str]]]]] = None,
) -> None:
    scores_groups = SCORES_GROUPS[score_groups_str]
    additional_scores_list = (
        additional_score_groups if additional_score_groups is not None else []
    )
    for group_name, (printable_group_name, group_scores) in (
        list(scores_groups.items()) + additional_scores_list
    ):
        n_cols = 3 if len(group_scores) > 3 else len(group_scores)
        n_rows = int(len(group_scores) // n_cols + 1 * (len(group_scores) % n_cols > 0))
        fig = plt.figure(figsize=((n_cols + 1) * 5, n_rows * 5))
        for n, score in enumerate(group_scores):
            ax = fig.add_subplot(n_rows, n_cols, n + 1)
            sns.boxplot(
                data=results_df,
                x="algo_name",
                y=-1 * results_df.loc[:, f"mean_test_{score}"]
                if score == "davies_bouldin"
                else f"mean_test_{score}",
                hue="results_tag"
                if results_df.loc[:, "results_tag"].unique().shape[0] > 1
                else None,
                ax=ax,
                order=ALGO_ORDER,
            )
            ax.set_title(
                ("-1 * " if score == "davies_bouldin" else "")
                + SCORES_PRETTY_NAMES[score]
            )
            ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
            box_patches = [patch for patch in ax.patches if type(patch) == PathPatch]
            lines_per_boxplot = len(ax.lines) // len(box_patches)
            for i, patch in enumerate(box_patches):
                face_col = patch.get_facecolor()
                patch.set_edgecolor(face_col)
                for line in ax.lines[
                    i * lines_per_boxplot : (i + 1) * lines_per_boxplot
                ]:
                    line.set_color(face_col)
                    line.set_mfc(face_col)  # facecolor of fliers
                    line.set_mec(face_col)  # edgecolor of fliers
            ax.legend([], [], frameon=False)
            ax.set_xlabel("")
            if score in ["silhouette", "adjusted_rand_index"]:
                ax.hlines(
                    [0], *ax.get_xlim(), linestyles="dashed", colors="gray", alpha=0.5
                )

        if printable_group_name is not None:
            fig.suptitle(
                f"Mean{printable_group_name}scores for differnt hyperparameters"
            )
        plt.legend(loc="lower left", bbox_to_anchor=(1.05, 0.4))
        plt.tight_layout()
        fig.savefig(
            save_path.parent
            / f"{save_path.stem}{'_' + group_name if group_name != '' else ''}{save_path.suffix}"
        )
        plt.close()


def boxplot_scores_per_best_config(
    results_df: pd.DataFrame,
    save_path: Path,
    score_groups_str: str = "none",
    static_score: Optional[str] = None,
    log_scale: bool = False,
    median_label: str = "none",
) -> None:
    scores_groups = SCORES_GROUPS[score_groups_str]
    for group_name, (printable_group_name, group_scores) in scores_groups.items():
        n_cols = 3 if len(group_scores) > 3 else len(group_scores)
        n_rows = int(len(group_scores) // n_cols + 1 * (len(group_scores) % n_cols > 0))
        fig = plt.figure(figsize=((n_cols + 1) * 5, n_rows * 5))
        for n, score in enumerate(group_scores):
            transform_func = SCORES[score]
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
                .filter(
                    regex=f"(split[0-9]+_test_{score if static_score is None else static_score}"
                    + "|algo_name|results_tag)"
                )
                .melt(id_vars=["algo_name", "results_tag"])
            )
            y = (
                -1 * max_results_df.loc[:, "value"]
                if static_score is None and score == "davies_bouldin"
                else max_results_df.loc[:, "value"]
            )
            ax = fig.add_subplot(n_rows, n_cols, n + 1)
            box_plot = sns.boxplot(
                data=max_results_df,
                x="algo_name",
                y=y,
                hue="results_tag"
                if max_results_df.loc[:, "results_tag"].unique().shape[0] > 1
                else None,
                palette=PALETTE,
                ax=ax,
                saturation=1,
                order=ALGO_ORDER,
            )
            if median_label != "none":
                medians = y.groupby(max_results_df.loc[:, "algo_name"]).median()
                y_median = y.median()
                vertical_offset = y_median * 0.1
                for xtick, xlabel_text in zip(
                    box_plot.get_xticks(), ax.get_xticklabels()
                ):
                    xlabel = xlabel_text.get_text()
                    box_plot.text(
                        xtick,
                        medians.loc[xlabel]
                        + (
                            vertical_offset
                            if medians.loc[xlabel] < y_median
                            else -2 * vertical_offset
                        ),
                        f"{medians.loc[xlabel]:.3f}"
                        if median_label == "float"
                        else f"{int(medians.loc[xlabel])}",
                        horizontalalignment="center",
                        size="small",
                        color="k",
                        weight="semibold",
                    )
            if log_scale:
                ax.set_yscale("log")
            box_patches = [patch for patch in ax.patches if type(patch) == PathPatch]
            lines_per_boxplot = len(ax.lines) // len(box_patches)
            for i, patch in enumerate(box_patches):
                face_col = patch.get_facecolor()
                patch.set_edgecolor(face_col)
                for line in ax.lines[
                    i * lines_per_boxplot : (i + 1) * lines_per_boxplot
                ]:
                    line.set_color(face_col)
                    line.set_mfc(face_col)  # facecolor of fliers
                    line.set_mec(face_col)  # edgecolor of fliers

            ax.set_title(
                (
                    ("-1 * " if score == "davies_bouldin" else "")
                    + SCORES_PRETTY_NAMES[score]
                )
                if static_score is None
                else SCORES_PRETTY_NAMES[static_score]
            )
            ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
            ax.set_xlabel("")
            ax.legend([], [], frameon=False)
            if score in ["silhouette", "adjusted_rand_index"]:
                ax.hlines(
                    [0], *ax.get_xlim(), linestyles="dashed", colors="gray", alpha=0.5
                )

        if printable_group_name is not None:
            fig.suptitle(
                f"Results for best hyperparameters for each{printable_group_name}score"
            )
        plt.legend(
            loc="lower left",
            bbox_to_anchor=(1.05, 0.4),
        )
        plt.tight_layout()
        fig.savefig(
            save_path.parent
            / f"{save_path.stem}{'_' + group_name if group_name != '' else ''}{save_path.suffix}"
        )
        plt.close()


def get_best_results_table(
    results_df: pd.DataFrame, static_score: Optional[str] = None
) -> pd.DataFrame:
    best_results_table = pd.DataFrame()
    for score, transform_func in SCORES.items():
        max_ids = (
            results_df.groupby(["algo_name"])[f"mean_test_{score}"].transform(
                transform_func,
            )
            == results_df.loc[:, f"mean_test_{score}"]
        )
        max_results = (
            (
                results_df.loc[
                    max_ids,
                    [
                        "algo_name",
                        f"mean_test_{score if static_score is None else static_score}",
                    ],
                ]
                .drop_duplicates()
                .set_index("algo_name")
            )
            .loc[:, f"mean_test_{score if static_score is None else static_score}"]
            .rename(score)
        )
        best_results_table = pd.concat([best_results_table, max_results], axis=1)
    best_results_table.columns = best_results_table.columns.map(SCORES_PRETTY_NAMES)
    best_results_table = pd.concat(
        [
            pd.concat(
                {group_name: best_results_table.loc[[algo_name]]},
                names=["algo_group"],
            )
            for algo_name, group_name in ALGO_GROUPS.items()
        ]
    )
    best_results_table.index.names = ["", "Algorithm"]
    return best_results_table


def get_best_labels_per_score_per_algo(
    results_df: pd.DataFrame,
    data_df: pd.DataFrame,
) -> dict[str, dict[str, pd.Series]]:
    labels_per_score_per_algo = {}
    for score, transform_func in tqdm(SCORES.items()):
        labels_per_score_per_algo[score] = {}
        best_ids = (
            results_df.groupby("algo_name")[f"mean_test_{score}"].transform(
                transform_func,
            )
            == results_df.loc[:, f"mean_test_{score}"]
        )
        best_results_params = results_df.loc[best_ids, "params"].map(lambda x: eval(x))

        for param_dict in best_results_params:
            pipeline = get_pipeline_from_params(param_dict)
            cluster_algo = param_dict["cluster_algo"]
            labels_per_score_per_algo[score][
                cluster_algo.__class__.__name__
            ] = pd.Series(
                pipeline.fit_predict(data_df.values), index=data_df.index
            ).map(
                str
            )

    return labels_per_score_per_algo


def embedding_plot_per_best_config(
    labels_per_score_per_algo: dict[str, dict[str, pd.Series]],
    embedding: pd.DataFrame,
    ground_truth: Optional[pd.Series],
    save_dir: Path,
) -> None:
    for score, labels_per_algo in tqdm(labels_per_score_per_algo.items()):
        score_dir = save_dir / score
        score_dir.mkdir(exist_ok=True)
        for algo_name, labels in labels_per_algo.items():
            plt.figure(figsize=(10, 5))
            sns.scatterplot(
                data=embedding,
                x="x",
                y="y",
                hue=labels,
            )
            plt.title(f"Best {algo_name} clustering on {SCORES_PRETTY_NAMES[score]}")
            plt.legend(
                loc="lower left",
                bbox_to_anchor=(1.05, 0.4),
            )
            plt.tight_layout()
            plt.savefig(
                score_dir
                / f"Embedding_best_hyperparameters_{score}_{algo_name.replace(' ', '_')}.png"
            )
            plt.close()
    if ground_truth is not None:
        plt.figure(figsize=(10, 5))
        sns.scatterplot(
            data=embedding,
            x="x",
            y="y",
            hue=ground_truth.map(str),
        )
        plt.title("External labels")
        plt.legend(
            loc="lower left",
            bbox_to_anchor=(1.05, 0.4),
        )
        plt.tight_layout()
        plt.savefig(save_dir / f"Embedding_best_hyperparameters_external_label.png")
        plt.close()


def joined_results_dir(results_dirs: list[Path]):
    return results_dirs[0].parent / "_vs_".join([r.name for r in results_dirs])


def main(
    results_dirs: list[Path],
    datasets: list[str],
    skip_embedding: bool = False,
    latex_float_precision: int = 3,
    latex_cluster_count_precision: int = 1,
    score_groups: str = "none",
    skip_table: bool = False,
    scaler: str = "min-max",
) -> None:
    datasets_dict = get_datasets_dict(scaler)
    for dataset in datasets:
        results_dfs = []
        for i, results_dir in enumerate(results_dirs):
            results_dfs.append(
                pd.read_csv(
                    results_dir / f"{dataset}_grid_search.csv", index_col=0
                ).reset_index()
            )
            results_dfs[i]["results_tag"] = RESULTS_NAMES_DICT.get(
                results_dir.name, results_dir.name
            )
        results_df = pd.concat(results_dfs)
        results_df.algo_name = results_df.algo_name.map(ALGO_PRETTY_NAMES)

        output_dir = joined_results_dir(results_dirs)
        output_dir.mkdir(exist_ok=True)

        if not skip_table:
            best_results_table = get_best_results_table(results_df)
            best_results_table.to_csv(output_dir / f"{dataset}_best_results_table.csv")
            best_results_table.style.format(precision=latex_float_precision).to_latex(
                output_dir / f"{dataset}_best_results_table.tex"
            )

            best_results_counts_table = get_best_results_table(
                results_df, static_score=CLUSTERS_COUNT_STR
            )
            best_results_counts_table.columns = best_results_table.columns
            best_results_counts_table.to_csv(
                output_dir / f"{dataset}_best_results_{CLUSTERS_COUNT_STR}_table.csv"
            )
            best_results_counts_table.style.format(
                precision=latex_cluster_count_precision,
            ).to_latex(
                output_dir / f"{dataset}_best_results_{CLUSTERS_COUNT_STR}_table.tex"
            )

        boxplot_mean_scores_per_algo(
            results_df,
            output_dir / f"{dataset}_mean_score_per.png",
            score_groups,
            additional_score_groups=[
                (CLUSTERS_COUNT_STR, (None, [CLUSTERS_COUNT_STR]))
            ],
        )

        boxplot_scores_per_best_config(
            results_df,
            output_dir / f"{dataset}_best_config_scores.png",
            score_groups,
        )

        boxplot_scores_per_best_config(
            results_df,
            output_dir / f"{dataset}_best_config_scores_{CLUSTERS_COUNT_STR}.png",
            score_groups,
            static_score=CLUSTERS_COUNT_STR,
            median_label="int",
        )

        if not skip_embedding and len(results_dirs) == 1:
            data_df, ground_truth = datasets_dict[dataset]()

            labels_per_score_per_algo = get_best_labels_per_score_per_algo(
                results_df,
                data_df,
            )

            if TSNE_EMBEDDING_FILE[dataset].exists():
                embedding = pd.read_csv(TSNE_EMBEDDING_FILE[dataset], index_col=0)
            else:
                embedding = pd.DataFrame(
                    TSNE(
                        n_components=2,
                        random_state=SEED,
                        perplexity=30,
                        metric="manhattan",
                    ).fit_transform(data_df),
                    index=data_df.index,
                    columns=["x", "y"],
                )
                embedding.to_csv(TSNE_EMBEDDING_FILE[dataset])

            tsne_dir = results_dir / f"{dataset}_tsne"
            tsne_dir.mkdir(exist_ok=True)

            embedding_plot_per_best_config(
                labels_per_score_per_algo,
                embedding,
                ground_truth,
                tsne_dir,
            )

            if PCA_EMBEDDING_FILE[dataset].exists():
                embedding = pd.read_csv(PCA_EMBEDDING_FILE[dataset], index_col=0)
            else:
                embedding = pd.DataFrame(
                    PCA(n_components=2, random_state=SEED).fit_transform(data_df),
                    index=data_df.index,
                    columns=["x", "y"],
                )
                embedding.to_csv(PCA_EMBEDDING_FILE[dataset])

            pca_dir = results_dir / f"{dataset}_pca"
            pca_dir.mkdir(exist_ok=True)

            embedding_plot_per_best_config(
                labels_per_score_per_algo,
                embedding,
                ground_truth,
                pca_dir,
            )

            if UMAP_EMBEDDING_FILE[dataset].exists():
                embedding = pd.read_csv(UMAP_EMBEDDING_FILE[dataset], index_col=0)
            else:
                embedding = pd.DataFrame(
                    UMAP(n_components=2, random_state=SEED).fit_transform(data_df),
                    index=data_df.index,
                    columns=["x", "y"],
                )
                embedding.to_csv(UMAP_EMBEDDING_FILE[dataset])

            umap_dir = results_dir / f"{dataset}_umap"
            umap_dir.mkdir(exist_ok=True)

            embedding_plot_per_best_config(
                labels_per_score_per_algo,
                embedding,
                ground_truth,
                umap_dir,
            )


if __name__ == "__main__":
    argparser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    argparser.add_argument(
        "results_dirs",
        type=path_list_parse,
        help="path(s) to directories with results, separated by coma (',')",
    )
    argparser.add_argument(
        "--datasets",
        type=lambda x: x.split(","),
        default=list(get_datasets_dict().keys()),
        help="names of datasets for which results are processed, separated by coma (',')",
    )
    argparser.add_argument(
        "--skip_embedding",
        default=False,
        action="store_true",
        help="flags if embedding plots should be skipped",
    )
    argparser.add_argument(
        "--latex_float_precision",
        default=3,
        help="precision level for floats (except cluster count means) in latex version of results table",
    )
    argparser.add_argument(
        "--latex_cluster_count_precision",
        default=1,
        help="precision level for cluster counts means in latex version of results table",
    )
    argparser.add_argument(
        "--score_groups",
        default="",
        choices=["", "ground-truth", "individual"],
        help="method for grouping scores on plots",
    )
    argparser.add_argument(
        "--skip_table",
        default=False,
        action="store_true",
        help="flags if the results table creation should be skipped",
    )
    argparser.add_argument(
        "--scaler",
        default="min-max",
        choices=["min-max", "none", "pca", "quantile", "standard"],
        help="select scaler used for data preprocessing",
    )
    kwargs = dict(argparser.parse_args()._get_kwargs())
    main(**kwargs)
