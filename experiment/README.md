# Experiments summary

Results (with plots), for each preprocessing are available in directories:
* Z-score normalization: [results_2023_05_24_1020_StandardScaler](./results_2023_05_24_1020_StandardScaler/)
* min-max normalization: [results_2023_05_24_2340_MinMaxScaler](./results_2023_05_24_2340_MinMaxScaler/)
* quantile normalization: [results_2023_05_24_1508_QuantileTransformer](./results_2023_05_24_1508_QuantileTransformer/)
* Z-score + PCA: [results_2023_05_23_1732_PCA](./results_2023_05_23_1732_PCA/)

Plots and table for all results together are availabel in [results_2023_05_23_1732_PCA_vs_results_2023_05_24_1020_StandardScaler_vs_results_2023_05_24_1508_QuantileTransformer_vs_results_2023_05_24_2340_MinMaxScaler](./results_2023_05_23_1732_PCA_vs_results_2023_05_24_1020_StandardScaler_vs_results_2023_05_24_1508_QuantileTransformer_vs_results_2023_05_24_2340_MinMaxScaler/).

## Results
GEMLER dataset. Mean score of the best configuration for that score for each algorithm. Top result for each score is marked in bold. ($SC$ - Silhouette Coeff.; $CH$ - Calinski-Harabasz Index; $DB$ - Davies-Bouldin Index; $ARI$ - Adjusted Rand Index; $AMI$ - Adjusted Mutual Info.):

|                    |                      | $SC$           | $CH$             | $DB$           | $ARI$          | $AMI$          |
|--------------------|----------------------|----------------|------------------|----------------|----------------|----------------|
|                    | Algorithm            |                |                  |                |                |                |
| Partition-based    | K-Means              | 0.086          | **147.704** | 2.868          | 0.452          | 0.526          |
|                    | K-Medoids            | 0.195          | 116.901          | 1.610          | 0.420          | 0.462          |
|                    | Affinity Propagation | 0.046          | 29.656           | 0.344          | 0.197          | 0.434          |
| Hierarchical       | AHC                  | **0.413** | 131.146          | 0.567          | 0.488          | 0.570          |
|                    | BIRCH                | 0.082          | 131.146          | **0.093** | 0.488          | 0.570          |
| Distribution-based | Gaussian Mixture     | 0.044          | 21.124           | 7.255          | 0.069          | 0.113          |
| Density-based      | DBSCAN               | 0.253          | 18.792           | 1.670          | 0.007          | 0.035          |
|                    | OPTICS               | 0.004          | 4.289            | 1.560          | 0.004          | 0.046          |
| Model-based        | SOM                  | 0.085          | 145.895          | 3.154          | 0.451          | 0.492          |
| Graph-based        | Spectral Clustering  | 0.254          | 111.043          | 0.614          | **0.547** | **0.615** |


GEMLER dataset. Mean number of clusters of the best configuration for each score for each algorithm. Count for top result for each score is marked in bold. ($SC$ - Silhouette Coeff.; $CH$ - Calinski-Harabasz Index; $DB$ - Davies-Bouldin Index; $ARI$ - Adjusted Rand Index; $AMI$ - Adjusted Mutual Info.):
|                    |                      | $SC$         | $CH$         | $DB$            | $ARI$        | $AMI$        |
|--------------------|----------------------|--------------|--------------|-----------------|--------------|--------------|
|                    | Algorithm            |              |              |                 |              |              |
| Partition-based    | K-Means              | 2.0          | **2.0** | 10.0            | 8.0          | 10.0         |
|                    | K-Medoids            | 2.0          | 2.0          | 2.0             | 6.0          | 8.0          |
|                    | Affinity Propagation | 43.0         | 5.4          | 1478.0          | 33.0         | 43.0         |
| Hierarchical       | AHC                  | **2.0** | 2.0          | 3.0             | 8.0          | 6.0          |
|                    | BIRCH                | 12.0         | 2.0          | **1486.0** | 8.0          | 6.0          |
| Distribution-based | Gaussian Mixture     | 2.0          | 3.0          | 7.0             | 7.0          | 7.0          |
| Density-based      | DBSCAN               | 2.0          | 2.0          | 10.0            | 10.0         | 10.0         |
|                    | OPTICS               | 2.0          | 11.0         | 11.0            | 24.0         | 24.0         |
| Model-based        | SOM                  | 2.0          | 2.0          | 2.0             | 8.0          | 8.0          |
| Graph-based        | Spectral Clustering  | 2.0          | 2.0          | 2.0             | **5.0** | **5.0** |


METABRIC dataset. Mean scores of the best configuration for that score for each algorithm. Top result for each score is marked in bold. ($SC$ - Silhouette Coeff.; $CH$ - Calinski-Harabasz Index; $DB$ - Davies-Bouldin Index; $ARI$ - Adjusted Rand Index; $AMI$ - Adjusted Mutual Info.):
|                    |                      | $SC$           | $CH$            | $DB$           | $ARI$          | $AMI$          |
|--------------------|----------------------|----------------|-----------------|----------------|----------------|----------------|
|                    | Algorithm            |                |                 |                |                |                |
| Partition-based    | K-Means              | 0.066          | **97.015** | 3.593          | 0.257          | 0.380          |
|                    | K-Medoids            | 0.135          | 83.165          | 1.095          | 0.206          | 0.332          |
|                    | Affinity Propagation | 0.023          | 15.209          | **0.481** | 0.162          | 0.277          |
| Hierarchical       | AHC                  | 0.163          | 79.560          | 0.731          | 0.276          | 0.373          |
|                    | BIRCH                | 0.078          | 13.451          | 2.277          | 0.072          | 0.129          |
| Distribution-based | Gaussian Mixture     | 0.000          | 18.211          | 6.567          | 0.106          | 0.134          |
| Density-based      | DBSCAN               | **0.280** | 66.543          | 1.870          | 0.059          | 0.134          |
|                    | OPTICS               | 0.071          | 9.867           | 1.530          | 0.003          | 0.012          |
| Model-based        | SOM                  | 0.039          | 85.505          | 4.079          | 0.227          | 0.346          |
| Graph-based        | Spectral Clustering  | 0.198          | 81.915          | 0.670          | **0.299** | **0.427** |

METABRIC dataset. Mean number of clusters of the best configuration for each score for each algorithm. Count for top result for each score is marked in bold. ($SC$ - Silhouette Coeff.; $CH$ - Calinski-Harabasz Index; $DB$ - Davies-Bouldin Index; $ARI$ - Adjusted Rand Index; $AMI$ - Adjusted Mutual Info.):
|                    |                      | $SC$         | $CH$         | $DB$            | $ARI$         | $AMI$        |
|--------------------|----------------------|--------------|--------------|-----------------|---------------|--------------|
|                    | Algorithm            |              |              |                 |               |              |
| Partition-based    | K-Means              | 2.0          | **2.0** | 3.0             | 4.0           | 8.0          |
|                    | K-Medoids            | 2.0          | 3.0          | 2.0             | 3.0           | 3.0          |
|                    | Affinity Propagation | 5.2          | 33.0         | **2086.0** | 15.8          | 33.0         |
| Hierarchical       | AHC                  | 2.0          | 2.0          | 2.0             | 4.0           | 3.0          |
|                    | BIRCH                | 2.0          | 2.0          | 2.0             | 14.0          | 14.0         |
| Distribution-based | Gaussian Mixture     | 2.0          | 5.0          | 6.0             | 7.0           | 7.0          |
| Density-based      | DBSCAN               | **2.0** | 2.0          | 2.0             | 2.0           | 2.0          |
|                    | OPTICS               | 2.0          | 2.0          | 2.0             | 3.2           | 3.2          |
| Model-based        | SOM                  | 2.0          | 2.0          | 4.0             | 6.0           | 10.0         |
| Graph-based        | Spectral Clustering  | 3.8          | 3.0          | 5.6             | **12.0** | **3.0** |


### Main conclusions:
* Evaluated popular internal measures of clustering quality work poorly for genomic data. One should either use external measures with some information about groups in the data or develop a custom internal measure specific to the genomic data.
 * When using standard internal measures, the count of clusters discovered by the best configuration can serve as a proxy of the clusterings informational value &mdash; too few clusters (2-3) mean that probably nothing interesting was discovered, too many clusters (almost as many as datapoints) mean that no new information was discovered.
* If standard internal measures need to be applied, then the best choice is simple clustering algorithms, especially K-Means.
* If external measures are used or other measures that can be trusted on the genomic data, then Spectral Clustering is the best solution.
*  Preprocessing techniques have a significant impact on the quality of the results, and they have to be individually chosen for the dataset and the algorithm. (PCA usually does a poor job on the genomic data)