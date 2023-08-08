# Clustering algorithms on genomic data
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.8136372.svg)](https://doi.org/10.5281/zenodo.8136372)

This is a repository with source code and results of experiments for my master's thesis: _CLUSTERING METHODS FOR GENOMIC DATA_. The purpose of the study was to evaluate popular clustering methods and quality measures on the genomic data. The results serve as basic guidance for researchers looking to choose the best method for clustering on genomic data.

## Experimental setup

### Datasets
In the study, I have used two microarray datasets:
* GEMLeR (Gene Expression Machine Learning Repository)
    * dataset of gene expression levels in cancer cells from nine different tissues,
    * 1488 samples, 7888 attributes (genes),
    * `Gregor Stiglic and Peter Kokol. Stability of ranked gene lists in large microarray analysis studies.
`_`Journal of biomedicine and biotechnology`_`, 2010, 2010.`

* METABRIC (The Molecular Taxonomy of Breast Cancer International
Consortium)
    * dataset of gene expression levels in cancer cells from six subtypes of breast cancer,
    * 2106 samples, 20000 attributes (genes),
    * `Christina Curtis, Sohrab P Shah, Suet-Feung Chin, Gulisa Turashvili, Oscar M Rueda, Mark J
Dunning, Doug Speed, Andy G Lynch, Shamith Samarajiwa, Yinyin Yuan, et al. The genomic and
transcriptomic architecture of 2,000 breast tumours reveals novel subgroups. `_`Nature`_`, 486(7403):
346–352, 2012`

Instructions for downloading the above datasets and adding new datasets are explained in [datasets/README.md](./datasets/README.md).

### Clustering algorithms
The study evaluated the follwing clustering algorithms:
* partitioning:
    * K-Means
    * K-Medoids
    * Affinity Propagation
* hierarchical:
    * AHC (Agglomerative Hierarchical Clustering)
    * BIRCH (Balanced Iterative Reducing and Clustering using Hierarchies)
* density-based:
    * DBSCAN (Density-Based Spatial Clustering of Applications with Noise)
    * OPTICS (Ordering Points To Identify the Clustering Structure)
* based on sample distribution:
    * Gaussian Mixtures
* based on a model:
    * SOM (Self Organising Maps)
* based on graph-theory
    * Spectral Clustering

### Quality measures
The algorithms were evaluated using five measures:
* internal:
    * Silhouette Coefficient
    * Calinski-Harabasz Index
    * Davies-Bouldin Index
* external:
    * Adjusted Rand Index (ARI)
    * Adjusted Mutual Information (AMI)


### Preprocessing
Additionally, four methods of preprocessing were considered:
* Z-score normalization ($\frac{x - \mu}{\sigma}$)
* min-max normalization ($\frac{x - x_{min}}{x_{max} - x_{min}}$)
* quantile normalization (transformation to form normal distribution from the data)
* Z-score + PCA dimensionality reduction

## Environment configuration
To run those experiments, you need Python 3.9 environment. You can use the prepared docker container or your own environment, for example, conda env.

To build and run the docker container, you will need `docker-compose`, then you can use the following commands:
```bash
docker-compose build
docker-compose run --rm env
```
This will create a container with the ready environment (Python 3.9 and necessary packages).

If you would like to use the conda env instead, do the following:
```bash
conda create -n clustering-genomic-data python=3.9
conda activate clustering-genomic-data
pip install -r requirements.txt
```
This will create a conda env with all necessary packages.

## Running experiments

Experiments are run using the Python script `run_experiments.py`:
```bash
usage: run_experiment.py [-h] [--scaler {min-max,pca,quantile,standard}] [--results_dir_tag RESULTS_DIR_TAG] [--experiment_dir EXPERIMENT_DIR] [--repeats REPEATS] [--shuffle_each_repeat]

optional arguments:
  -h, --help            show this help message and exit
  --scaler {min-max,pca,quantile,standard}
                        select scaler used for data preprocessing (default: min-max)
  --results_dir_tag RESULTS_DIR_TAG
                        tag added as suffix to the results directory name (default: )
  --experiment_dir EXPERIMENT_DIR
                        experiment directory, inside of which results directory will be created (default: ./experiment)
  --repeats REPEATS     number of repeats of the evaluation (default: 5)
  --shuffle_each_repeat
                        flags if the order of samples in the dataset should be shuffled in each repeat (default: False)
```

To reproduce experiments from the study, run the following commands (in a proper environment):
```bash
cd experiment
python run_experiment.py --scaler pca --results_dir_tag PCA --repeats 5 --shuffle_each_repeat
python run_experiment.py --scaler standard --results_dir_tag StandardScaler --repeats 5 --shuffle_each_repeat
python run_experiment.py --scaler quantile --results_dir_tag QuantileTransformer --repeats 5 --shuffle_each_repeat
python run_experiment.py --scaler min-max --results_dir_tag MinMaxScaler --repeats 5 --shuffle_each_repeat
```

## Experiments summary

Python script `results_summary.py` processes the results. It creates:
* a table with the mean results for the best configuration of each algorithm for each score,
* a table with the mean cluster count for the best configuration of each algorithm for each score,
* plots for mean results of best algorithms for each score,
* plots for the mean results of all configurations of each algorithm for each score,
* embedding plots (t-SNE, UMAP, PCA) with the clusterings of the best configuration of each algorithm for each score.
The study's results and their summary are presented in [experiment/README.md](./experiment/README.md).

Usage:
```bash
usage: results_summary.py [-h] [--datasets DATASETS] [--skip_embedding] [--latex_float_precision LATEX_FLOAT_PRECISION] [--latex_cluster_count_precision LATEX_CLUSTER_COUNT_PRECISION]
                          [--score_groups {,ground-truth,individual}] [--skip_table] [--scaler {min-max,none,pca,quantile,standard}]
                          results_dirs

positional arguments:
  results_dirs          path(s) to directories with results, separated by coma (',')

optional arguments:
  -h, --help            show this help message and exit
  --datasets DATASETS   names of datasets for which results are processed, separated by coma (',') (default: ['GEMLER', 'METABRIC'])
  --skip_embedding      flags if embedding plots should be skipped (default: False)
  --latex_float_precision LATEX_FLOAT_PRECISION
                        precision level for floats (except cluster count means) in latex version of results table (default: 3)
  --latex_cluster_count_precision LATEX_CLUSTER_COUNT_PRECISION
                        precision level for cluster counts means in latex version of results table (default: 1)
  --score_groups {,ground-truth,individual}
                        method for grouping scores on plots (default: )
  --skip_table          flags if the results table creation should be skipped (default: False)
  --scaler {min-max,none,pca,quantile,standard}
                        select scaler used for data preprocessing (default: min-max)
```

To summarize results from the study run the following commands:
```bash
cd experiment

python results_summary.py results_2023_05_23_1732_PCA --score_groups individual --skip_table --scaler pca
python results_summary.py results_2023_05_24_1020_StandardScaler --score_groups individual --skip_table --scaler standard
python results_summary.py results_2023_05_24_1508_QuantileTransformer --score_groups individual --skip_table --scaler quantile
python results_summary.py results_2023_05_24_2340_MinMaxScaler --score_groups individual --skip_table --scaler min-max

python results_summary.py \
results_2023_05_23_1732_PCA_WHOLE_DATA,results_2023_05_24_1020_StandardScaler_WHOLE_DATA,results_2023_05_24_1508_QuantileTransformer_WHOLE_DATA,results_2023_05_24_2340_MinMaxScaler_WHOLE_DATA \
--score_groups individual
```
