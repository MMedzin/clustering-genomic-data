from argparse import ArgumentParser
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
from matplotlib.patches import PathPatch
from matplotlib.legend import _get_legend_handles_labels
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
from utils import (
    SEED,
    load_gemler_data_normed,
    load_metabric_data_normed,
    get_pipeline_from_params,
)

DATASETS = {
    "GEMLER": load_gemler_data_normed(),
    "METABRIC": load_metabric_data_normed(),
}

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

ALGO_GROUPS = {
    "KMeans": "Partition-based",
    "KMedoids": "Partition-based",
    "AffinityPropagation": "Partition-based",
    "AgglomerativeClustering": "Hierarchical",
    "Birch": "Hierarchical",
    "GaussianMixture": "Distribution-based",
    "DBSCAN": "Density-based",
    "OPTICS": "Density-based",
    "SOM": "Model-based",
    "SpectralClustering": "Graph-based",
}

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


def boxplot_scores_per_best_config(
    results_df: pd.DataFrame,
    save_path: Path,
    score_groups_str: str = "none",
    static_score: Optional[str] = None,
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
                else "value"
            )
            ax = fig.add_subplot(n_rows, n_cols, n + 1)
            sns.boxplot(
                data=max_results_df,
                x="algo_name",
                y=y,
                hue="results_tag"
                if max_results_df.loc[:, "results_tag"].unique().shape[0] > 1
                else None,
                palette=PALETTE,
                ax=ax,
                saturation=1,
            )
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


def get_best_results_table(results_df: pd.DataFrame) -> pd.DataFrame:
    best_results_table = pd.DataFrame()
    for score, transform_func in SCORES.items():
        max_ids = (
            results_df.groupby(["algo_name"])[f"mean_test_{score}"].transform(
                transform_func,
            )
            == results_df.loc[:, f"mean_test_{score}"]
        )
        max_results_df = (
            results_df.loc[max_ids, ["algo_name", f"mean_test_{score}"]]
            .drop_duplicates()
            .set_index("algo_name")
        )
        best_results_table = pd.concat([best_results_table, max_results_df], axis=1)
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
        n_cols = 3
        n_plots = len(labels_per_algo) + 1 * (ground_truth is not None)
        n_rows = int(n_plots // n_cols + 1 * (n_plots % n_cols > 0))
        fig = plt.figure(figsize=(n_rows * 20, n_cols * 5))

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
        fig.suptitle(
            f"Embedding visualization for best hyperparameters for {SCORES_PRETTY_NAMES[score]}"
        )
        plt.tight_layout()
        fig.savefig(save_dir / f"Embedding_best_hyperparameters_{score}.png")


def joined_results_dir(results_dirs: list[Path]):
    return results_dirs[0].parent / "_vs_".join([r.name for r in results_dirs])


def main(
    results_dirs: list[Path],
    datasets: list[str],
    skip_embedding: bool = False,
    latex_float_precision: int = 3,
    score_groups: str = "none",
    skip_table: bool = False,
) -> None:
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

        output_dir = joined_results_dir(results_dirs)
        output_dir.mkdir(exist_ok=True)

        if not skip_table:
            best_results_table = get_best_results_table(results_df)
            best_results_table.to_csv(output_dir / f"{dataset}_best_results_table.csv")
            best_results_table.style.format(precision=latex_float_precision).to_latex(
                output_dir / f"{dataset}_best_results_table.tex"
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
    argparser.add_argument("--latex_float_precision", default=3)
    argparser.add_argument(
        "--score_groups", default="", choices=["", "ground-truth", "individual"]
    )
    argparser.add_argument("--skip_table", default=False, action="store_true")
    kwargs = dict(argparser.parse_args()._get_kwargs())
    main(**kwargs)
