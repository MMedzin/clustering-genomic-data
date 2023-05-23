from __future__ import annotations

import logging
import time
from typing import Generator, Union

import pandas as pd
from sklearn.model_selection import (
    BaseCrossValidator,
    GridSearchCV,
    ParameterGrid,
    check_cv,
)
from sklearn.pipeline import Pipeline
from sklearn.metrics._scorer import check_scoring, _check_multimetric_scoring
from sklearn.utils.validation import indexable, _check_fit_params
from sklearn.utils.parallel import Parallel, delayed
from sklearn.base import is_classifier, clone
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
    y_pred = pd.Series(pipeline.fit_predict(X.values), index=X.index)
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
        parallel = Parallel(n_jobs=self.n_jobs)
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


# class ClusteringGridSearchCV(GridSearchCV):
#     def fit(self, X, y=None, *, groups=None, **fit_params):
#         estimator = self.estimator
#         refit_metric = "score"

#         if callable(self.scoring):
#             scorers = self.scoring
#         elif self.scoring is None or isinstance(self.scoring, str):
#             scorers = check_scoring(self.estimator, self.scoring)
#         else:
#             scorers = _check_multimetric_scoring(self.estimator, self.scoring)

#         X, y, groups = indexable(X, y, groups)
#         fit_params = _check_fit_params(X, fit_params)

#         cv_orig = check_cv(self.cv, y, classifier=is_classifier(estimator))
#         n_splits = cv_orig.get_n_splits(X, y, groups)

#         base_estimator = clone(self.estimator)

#         parallel = Parallel(n_jobs=self.n_jobs, pre_dispatch=self.pre_dispatch)

#         fit_and_score_kwargs = dict(
#             scorer=scorers,
#             fit_params=fit_params,
#             return_train_score=self.return_train_score,
#             return_n_test_samples=True,
#             return_times=True,
#             return_parameters=False,
#             error_score=self.error_score,
#             verbose=self.verbose,
#         )
#         results = {}
#         with parallel:
#             all_candidate_params = []
#             all_out = []
#             all_more_results = defaultdict(list)

#             def evaluate_candidates(candidate_params, cv=None, more_results=None):
#                 cv = cv or cv_orig
#                 candidate_params = list(candidate_params)
#                 n_candidates = len(candidate_params)

#                 if self.verbose > 0:
#                     print(
#                         "Fitting {0} folds for each of {1} candidates,"
#                         " totalling {2} fits".format(
#                             n_splits, n_candidates, n_candidates * n_splits
#                         )
#                     )

#                 out = parallel(
#                     delayed(_fit_and_score)(
#                         clone(base_estimator),
#                         X,
#                         y,
#                         train=train,
#                         test=test,
#                         parameters=parameters,
#                         split_progress=(split_idx, n_splits),
#                         candidate_progress=(cand_idx, n_candidates),
#                         **fit_and_score_kwargs,
#                     )
#                     for (cand_idx, parameters), (split_idx, (train, test)) in product(
#                         enumerate(candidate_params), enumerate(cv.split(X, y, groups))
#                     )
#                 )

#                 if len(out) < 1:
#                     raise ValueError(
#                         "No fits were performed. "
#                         "Was the CV iterator empty? "
#                         "Were there no candidates?"
#                     )
#                 elif len(out) != n_candidates * n_splits:
#                     raise ValueError(
#                         "cv.split and cv.get_n_splits returned "
#                         "inconsistent results. Expected {} "
#                         "splits, got {}".format(n_splits, len(out) // n_candidates)
#                     )

#                 _warn_or_raise_about_fit_failures(out, self.error_score)

#                 # For callable self.scoring, the return type is only know after
#                 # calling. If the return type is a dictionary, the error scores
#                 # can now be inserted with the correct key. The type checking
#                 # of out will be done in `_insert_error_scores`.
#                 if callable(self.scoring):
#                     _insert_error_scores(out, self.error_score)

#                 all_candidate_params.extend(candidate_params)
#                 all_out.extend(out)

#                 if more_results is not None:
#                     for key, value in more_results.items():
#                         all_more_results[key].extend(value)

#                 nonlocal results
#                 results = self._format_results(
#                     all_candidate_params, n_splits, all_out, all_more_results
#                 )

#                 return results
