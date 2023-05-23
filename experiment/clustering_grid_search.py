from __future__ import annotations

import logging
import numbers
import sys
import time
import warnings
from collections import defaultdict
from contextlib import suppress
from functools import partial
from itertools import product
from traceback import format_exc
from typing import Generator, Union

import numpy as np
import pandas as pd
from joblib import logger
from sklearn.base import clone, is_classifier
from sklearn.metrics._scorer import _check_multimetric_scoring, check_scoring
from sklearn.model_selection import (
    BaseCrossValidator,
    GridSearchCV,
    ParameterGrid,
    check_cv,
)
from sklearn.model_selection._validation import (
    _insert_error_scores,
    _warn_or_raise_about_fit_failures,
)
from sklearn.pipeline import Pipeline
from sklearn.utils.metaestimators import _safe_split
from sklearn.utils.parallel import Parallel, delayed
from sklearn.utils.validation import _check_fit_params, _num_samples, indexable
from utils import get_pipeline_from_params

# logger = logging.getLogger("grid_search")
# logger.setLevel(logging.INFO)
# log_output_handler = logging.StreamHandler(sys.stdout)
# log_output_handler.setFormatter(
#     logging.Formatter(
#         "[CV] %(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s", "%H:%M:%S"
#     )
# )
# logger.addHandler(log_output_handler)


# def run_single_evaluation(
#     id: int,
#     prefix: str,
#     params: dict,
#     pipeline: Pipeline,
#     X: pd.DataFrame,
#     y: pd.Series,
#     label_scorers: dict,
#     verbose: bool = True,
# ) -> pd.Series:
#     start = time.time()
#     y_pred = pd.Series(pipeline.fit_predict(X.values), index=X.index)
#     fit_time = time.time() - start
#     result_dict = {f"{prefix}_fit_time": fit_time}
#     for name, scorer in label_scorers.items():
#         try:
#             score = scorer(y_pred, X, y)
#         except ValueError:
#             score = None
#         result_dict[f"{prefix}_{name}"] = score
#     result_dict["params"] = params
#     if verbose:
#         logger.info(
#             "GridSearch done for split %s for params %d for pipeline %s",
#             prefix,
#             id,
#             str(pipeline),
#         )
#     return pd.Series(result_dict, name=id)


# class ClusteringGridSearchCV:
#     def __init__(
#         self,
#         cv: BaseCrossValidator,
#         param_grid: Union[dict, list[dict]],
#         label_scorers: dict,
#         n_jobs: int = -1,
#         verbose: bool = True,
#     ) -> None:
#         self.cv = cv
#         self.param_grid = param_grid
#         self.label_scorers = label_scorers
#         self.n_jobs = n_jobs
#         self.verbose = verbose
#         self.cv_results_ = None

#     def _iter_paramed_pipelines(
#         self,
#     ) -> tuple[Generator[Pipeline, None, None], dict]:
#         for params in ParameterGrid(self.param_grid):
#             yield get_pipeline_from_params(params), params

#     @staticmethod
#     def _merge_results(results: list[pd.Series]) -> pd.DataFrame:
#         results_dict = {}
#         for series in results:
#             series_name = series.name
#             for index, value in series.items():
#                 if index not in results_dict:
#                     results_dict[index] = {}
#                 results_dict[index][series_name] = value
#         return pd.DataFrame(results_dict)

#     @staticmethod
#     def _process_results_table(results_df: pd.DataFrame) -> pd.DataFrame:
#         names_to_calculate = (
#             results_df.filter(regex="split.+_test")
#             .columns.map(lambda x: "_".join(x.split("_")[2:]))
#             .unique()
#             .to_list()
#         )
#         for name in names_to_calculate:
#             name_cols = results_df.filter(regex=f"split.+_test_{name}")
#             results_df[f"mean_test_{name}"] = name_cols.mean(axis=1)
#             results_df[f"std_test_{name}"] = name_cols.std(axis=1)
#             results_df[f"rank_test_{name}"] = name_cols.mean(axis=1).rank()
#         return results_df

#     def fit(self, X: pd.DataFrame, y: pd.Series) -> ClusteringGridSearchCV:
#         parallel = Parallel(n_jobs=self.n_jobs)
#         with parallel:
#             results = parallel(
#                 delayed(run_single_evaluation)(
#                     pipeline_id,
#                     f"split{split_id}_test",
#                     params,
#                     pipeline,
#                     X.iloc[train],
#                     y.iloc[train],
#                     self.label_scorers,
#                     self.verbose,
#                 )
#                 for split_id, (train, _) in enumerate(self.cv.split(X, y))
#                 for pipeline_id, (pipeline, params) in enumerate(
#                     self._iter_paramed_pipelines()
#                 )
#             )
#         results_df = self._merge_results(results)
#         results_df = self._process_results_table(results_df)
#         self.cv_results_ = results_df.to_dict()
#         return self


class _MultimetricLabelScorer:
    def __init__(self, *, scorers, raise_exc=True):
        self._scorers = scorers
        self._raise_exc = raise_exc

    def __call__(self, labels, *args, **kwargs):
        """Evaluate predicted target values."""
        scores = {}

        for name, scorer in self._scorers.items():
            try:
                score = scorer(labels, *args, **kwargs)
                scores[name] = score
            except Exception as e:
                if self._raise_exc:
                    raise e
                else:
                    scores[name] = format_exc()

        return scores


def _label_score(labels, X_test, y_test, scorer, error_score="raise"):
    if isinstance(scorer, dict):
        # will cache method calls if needed. scorer() returns a dict
        scorer = _MultimetricLabelScorer(
            scorers=scorer, raise_exc=(error_score == "raise")
        )

    try:
        if y_test is None:
            scores = scorer(labels, X_test)
        else:
            scores = scorer(labels, X_test, y_test)
    except Exception:
        if isinstance(scorer, _MultimetricLabelScorer):
            # If `_MultimetricScorer` raises exception, the `error_score`
            # parameter is equal to "raise".
            raise
        else:
            if error_score == "raise":
                raise
            else:
                scores = error_score
                warnings.warn(
                    "Scoring failed. The score on this train-test partition for "
                    f"these parameters will be set to {error_score}. Details: \n"
                    f"{format_exc()}",
                    UserWarning,
                )

    # Check non-raised error messages in `_MultimetricScorer`
    if isinstance(scorer, _MultimetricLabelScorer):
        exception_messages = [
            (name, str_e) for name, str_e in scores.items() if isinstance(str_e, str)
        ]
        if exception_messages:
            # error_score != "raise"
            for name, str_e in exception_messages:
                scores[name] = error_score
                warnings.warn(
                    "Scoring failed. The score on this train-test partition for "
                    f"these parameters will be set to {error_score}. Details: \n"
                    f"{str_e}",
                    UserWarning,
                )

    error_msg = "scoring must return a number, got %s (%s) instead. (scorer=%s)"
    if isinstance(scores, dict):
        for name, score in scores.items():
            if hasattr(score, "item"):
                with suppress(ValueError):
                    # e.g. unwrap memmapped scalars
                    score = score.item()
            if not isinstance(score, numbers.Number):
                raise ValueError(error_msg % (score, type(score), name))
            scores[name] = score
    else:  # scalar
        if hasattr(scores, "item"):
            with suppress(ValueError):
                # e.g. unwrap memmapped scalars
                scores = scores.item()
        if not isinstance(scores, numbers.Number):
            raise ValueError(error_msg % (scores, type(scores), scorer))
    return scores


def _fit_and_score(
    estimator,
    X,
    y,
    scorer,
    train,
    test,
    verbose,
    parameters,
    fit_params,
    return_train_score=False,
    return_parameters=False,
    return_n_test_samples=False,
    return_times=False,
    return_estimator=False,
    split_progress=None,
    candidate_progress=None,
    error_score=np.nan,
):
    if not isinstance(error_score, numbers.Number) and error_score != "raise":
        raise ValueError(
            "error_score must be the string 'raise' or a numeric value. "
            "(Hint: if using 'raise', please make sure that it has been "
            "spelled correctly.)"
        )

    progress_msg = ""
    if verbose > 2:
        if split_progress is not None:
            progress_msg = f" {split_progress[0]+1}/{split_progress[1]}"
        if candidate_progress and verbose > 9:
            progress_msg += f"; {candidate_progress[0]+1}/{candidate_progress[1]}"

    if verbose > 1:
        if parameters is None:
            params_msg = ""
        else:
            sorted_keys = sorted(parameters)  # Ensure deterministic o/p
            params_msg = ", ".join(f"{k}={parameters[k]}" for k in sorted_keys)
    if verbose > 9:
        start_msg = f"[CV{progress_msg}] START {params_msg}"
        print(f"{start_msg}{(80 - len(start_msg)) * '.'}")

    # Adjust length of sample weights
    fit_params = fit_params if fit_params is not None else {}
    fit_params = _check_fit_params(X, fit_params, train)

    if parameters is not None:
        # clone after setting parameters in case any parameters
        # are estimators (like pipeline steps)
        # because pipeline doesn't clone steps in fit
        cloned_parameters = {}
        for k, v in parameters.items():
            cloned_parameters[k] = clone(v, safe=False)

        estimator = estimator.set_params(**cloned_parameters)

    start_time = time.time()

    X_train, y_train = _safe_split(estimator, X, y, train)
    X_test, y_test = _safe_split(estimator, X, y, test, train)

    result = {}
    try:
        if y_train is None:
            labels = estimator.fit_predict(X_train, **fit_params)
        else:
            labels = estimator.fit_predict(X_train, y_train, **fit_params)

    except Exception:
        # Note fit time as time until error
        fit_time = time.time() - start_time
        score_time = 0.0
        if error_score == "raise":
            raise
        elif isinstance(error_score, numbers.Number):
            if isinstance(scorer, dict):
                test_scores = {name: error_score for name in scorer}
                if return_train_score:
                    train_scores = test_scores.copy()
            else:
                test_scores = error_score
                if return_train_score:
                    train_scores = error_score
        result["fit_error"] = format_exc()
    else:
        result["fit_error"] = None

        fit_time = time.time() - start_time
        test_scores = _label_score(labels, X_test, y_test, scorer, error_score)
        score_time = time.time() - start_time - fit_time
        if return_train_score:
            train_scores = _label_score(labels, X_train, y_train, scorer, error_score)

    if verbose > 1:
        total_time = score_time + fit_time
        end_msg = f"[CV{progress_msg}] END "
        result_msg = params_msg + (";" if params_msg else "")
        if verbose > 2:
            if isinstance(test_scores, dict):
                for scorer_name in sorted(test_scores):
                    result_msg += f" {scorer_name}: ("
                    if return_train_score:
                        scorer_scores = train_scores[scorer_name]
                        result_msg += f"train={scorer_scores:.3f}, "
                    result_msg += f"test={test_scores[scorer_name]:.3f})"
            else:
                result_msg += ", score="
                if return_train_score:
                    result_msg += f"(train={train_scores:.3f}, test={test_scores:.3f})"
                else:
                    result_msg += f"{test_scores:.3f}"
        result_msg += f" total time={logger.short_format_time(total_time)}"

        # Right align the result_msg
        end_msg += "." * (80 - len(end_msg) - len(result_msg))
        end_msg += result_msg
        print(end_msg)

    result["test_scores"] = test_scores
    if return_train_score:
        result["train_scores"] = train_scores
    if return_n_test_samples:
        result["n_test_samples"] = _num_samples(X_test)
    if return_times:
        result["fit_time"] = fit_time
        result["score_time"] = score_time
    if return_parameters:
        result["parameters"] = parameters
    if return_estimator:
        result["estimator"] = estimator
    return result


class ClusteringGridSearchCV(GridSearchCV):
    def fit(self, X, y=None, *, groups=None, **fit_params):
        estimator = self.estimator

        if callable(self.scoring):
            scorers = self.scoring
        elif self.scoring is None or isinstance(self.scoring, str):
            scorers = check_scoring(self.estimator, self.scoring)
        else:
            scorers = _check_multimetric_scoring(self.estimator, self.scoring)

        X, y, groups = indexable(X, y, groups)
        fit_params = _check_fit_params(X, fit_params)

        cv_orig = check_cv(self.cv, y, classifier=is_classifier(estimator))
        n_splits = cv_orig.get_n_splits(X, y, groups)

        base_estimator = clone(self.estimator)

        parallel = Parallel(n_jobs=self.n_jobs, pre_dispatch=self.pre_dispatch)

        fit_and_score_kwargs = dict(
            scorer=scorers,
            fit_params=fit_params,
            return_train_score=self.return_train_score,
            return_n_test_samples=True,
            return_times=True,
            return_parameters=False,
            error_score=self.error_score,
            verbose=self.verbose,
        )
        results = {}
        with parallel:
            all_candidate_params = []
            all_out = []
            all_more_results = defaultdict(list)

            def evaluate_candidates(candidate_params, cv=None, more_results=None):
                cv = cv or cv_orig
                candidate_params = list(candidate_params)
                n_candidates = len(candidate_params)

                if self.verbose > 0:
                    print(
                        "Fitting {0} folds for each of {1} candidates,"
                        " totalling {2} fits".format(
                            n_splits, n_candidates, n_candidates * n_splits
                        )
                    )

                out = parallel(
                    delayed(_fit_and_score)(
                        clone(base_estimator),
                        X,
                        y,
                        train=train,
                        test=test,
                        parameters=parameters,
                        split_progress=(split_idx, n_splits),
                        candidate_progress=(cand_idx, n_candidates),
                        **fit_and_score_kwargs,
                    )
                    for (cand_idx, parameters), (split_idx, (train, test)) in product(
                        enumerate(candidate_params), enumerate(cv.split(X, y, groups))
                    )
                )

                if len(out) < 1:
                    raise ValueError(
                        "No fits were performed. "
                        "Was the CV iterator empty? "
                        "Were there no candidates?"
                    )
                elif len(out) != n_candidates * n_splits:
                    raise ValueError(
                        "cv.split and cv.get_n_splits returned "
                        "inconsistent results. Expected {} "
                        "splits, got {}".format(n_splits, len(out) // n_candidates)
                    )

                _warn_or_raise_about_fit_failures(out, self.error_score)

                # For callable self.scoring, the return type is only know after
                # calling. If the return type is a dictionary, the error scores
                # can now be inserted with the correct key. The type checking
                # of out will be done in `_insert_error_scores`.
                if callable(self.scoring):
                    _insert_error_scores(out, self.error_score)

                all_candidate_params.extend(candidate_params)
                all_out.extend(out)

                if more_results is not None:
                    for key, value in more_results.items():
                        all_more_results[key].extend(value)

                nonlocal results
                results = self._format_results(
                    all_candidate_params, n_splits, all_out, all_more_results
                )

                return results

            self._run_search(evaluate_candidates)

            # multimetric is determined here because in the case of a callable
            # self.scoring the return type is only known after calling
            first_test_score = all_out[0]["test_scores"]
            self.multimetric_ = isinstance(first_test_score, dict)

            # check refit_metric now for a callabe scorer that is multimetric
            if callable(self.scoring) and self.multimetric_:
                self._check_refit_for_multimetric(first_test_score)
                refit_metric = self.refit

        # Store the only scorer not as a dict for single metric evaluation
        self.scorer_ = scorers

        self.cv_results_ = results
        self.n_splits_ = n_splits

        return self
