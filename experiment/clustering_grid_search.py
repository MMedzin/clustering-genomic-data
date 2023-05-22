from __future__ import annotations

import logging
import time
from typing import Generator, Union

import pandas as pd
from joblib.parallel import Parallel, delayed
from sklearn.model_selection import BaseCrossValidator, ParameterGrid
from sklearn.pipeline import Pipeline
from utils import get_pipeline_from_params
import sys

logger = logging.getLogger("grid_search")
logger.setLevel(logging.INFO)
log_output_handler = logging.StreamHandler(sys.stdout)
log_output_handler.setFormatter(
    logging.Formatter(
        "[CV] %(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s", "%H:%M:%S"
    )
)
logger.addHandler(log_output_handler)


def run_single_evaluation(
    id: int,
    prefix: str,
    params: dict,
    pipeline: Pipeline,
    X: pd.DataFrame,
    y: pd.Series,
    label_scorers: dict,
    verbose: bool = True,
) -> pd.Series:
    start = time.time()
    y_pred = pipeline.fit_predict(X)
    fit_time = time.time() - start
    result_dict = {f"{prefix}_fit_time": fit_time}
    for name, scorer in label_scorers.items():
        try:
            score = scorer(y_pred, X, y)
        except ValueError:
            score = None
        result_dict[f"{prefix}_{name}"] = score
    result_dict["params"] = params
    if verbose:
        logger.info(
            "GridSearch done for split %s for params %d for pipeline %s",
            prefix,
            id,
            str(pipeline),
        )
    return pd.Series(result_dict, name=id)


class ClusteringGridSearchCV:
    def __init__(
        self,
        cv: BaseCrossValidator,
        param_grid: Union[dict, list[dict]],
        label_scorers: dict,
        n_jobs: int = -1,
        verbose: bool = True,
    ) -> None:
        self.cv = cv
        self.param_grid = param_grid
        self.label_scorers = label_scorers
        self.n_jobs = n_jobs
        self.verbose = verbose
        self.cv_results_ = None

    def _iter_paramed_pipelines(
        self,
    ) -> tuple[Generator[Pipeline, None, None], dict]:
        for params in ParameterGrid(self.param_grid):
            yield get_pipeline_from_params(params), params

    @staticmethod
    def _merge_results(results: list[pd.Series]) -> pd.DataFrame:
        results_dict = {}
        for series in results:
            series_name = series.name
            for index, value in series.items():
                if index not in results_dict:
                    results_dict[index] = {}
                results_dict[index][series_name] = value
        return pd.DataFrame(results_dict)

    @staticmethod
    def _process_results_table(results_df: pd.DataFrame) -> pd.DataFrame:
        names_to_calculate = (
            results_df.filter(regex="split.+_test")
            .columns.map(lambda x: "_".join(x.split("_")[2:]))
            .unique()
            .to_list()
        )
        for name in names_to_calculate:
            name_cols = results_df.filter(regex=f"split.+_test_{name}")
            results_df[f"mean_test_{name}"] = name_cols.mean(axis=1)
            results_df[f"std_test_{name}"] = name_cols.std(axis=1)
            results_df[f"rank_test_{name}"] = name_cols.mean(axis=1).rank()
        return results_df

    def fit(self, X: pd.DataFrame, y: pd.Series) -> ClusteringGridSearchCV:
        parallel = Parallel(n_jobs=self.n_jobs, backend="loky")
        with parallel:
            results = parallel(
                delayed(run_single_evaluation)(
                    pipeline_id,
                    f"split{split_id}_test",
                    params,
                    pipeline,
                    X.iloc[train],
                    y.iloc[train],
                    self.label_scorers,
                    self.verbose,
                )
                for split_id, (train, _) in enumerate(self.cv.split(X, y))
                for pipeline_id, (pipeline, params) in enumerate(
                    self._iter_paramed_pipelines()
                )
            )
        results_df = self._merge_results(results)
        results_df = self._process_results_table(results_df)
        self.cv_results_ = results_df.to_dict()
        return self
