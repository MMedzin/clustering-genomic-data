# Datasets

Two datasets were used in this study - GEMLeR and METABRIC. Instructions how to get neccessary files are explained below.

## GEMLeR
This dataset was sourced from [https://www.openml.org](https://www.openml.org). Five subsets of samples were used. They are stored in five separate `.arff` files and are available at:
* `AP_Breast_Ovary.arff`: [https://www.openml.org/search?type=data&status=active&id=1165](https://www.openml.org/search?type=data&status=active&id=1165)
* `AP_Colon_Kidney.arff`: [https://www.openml.org/search?type=data&status=active&id=1137](https://www.openml.org/search?type=data&status=active&id=1137)
* `AP_Endometrium_Prostate.arff`: [https://www.openml.org/search?type=data&status=active&id=1141](https://www.openml.org/search?type=data&status=active&id=1141)
* `AP_Omentum_Lung.arff`: [https://www.openml.org/search?type=data&status=active&id=1132](https://www.openml.org/search?type=data&status=active&id=1132)
* `AP_Prostate_Uterus.arff`: [https://www.openml.org/search?type=data&status=active&id=1131](https://www.openml.org/search?type=data&status=active&id=1131) (Prostate samples are duplicated in `AP_Endometrium_Prostate.arff` and `AP_Prostate_Uterus.arff`, but duplicates are dropped on loading)

Those files are already loaded to this repository, as they are relatively small.

Those datasets are merged to form one big dataset with 1488 samples and 7888 attributes (gene expression levels). Only genes common for all samples were left, to avoid missing values. The following table presents distribution of labels in the merged dataset:
| Tissue | Breast | Colon | Kidney | Ovary | Lung | Uterus | Omentum | Prostate | Endometrium |
|--------|--------|-------|--------|-------|------|--------|---------|----------|-------------|
| count  | 333    | 278   | 252    | 183   | 119  | 119    | 76      | 67       | 61          |



## METABRIC

This dataset was sourced from the [DeepType](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC8215925/) paper repository ([https://github.com/runpuchen/DeepType](https://github.com/runpuchen/DeepType)), where it is shared via google drive: [https://drive.google.com/file/d/1ao1zu3DS8GkYF-tHxpQ-1ev2psxXL-fx/view?usp=sharing](https://drive.google.com/file/d/1ao1zu3DS8GkYF-tHxpQ-1ev2psxXL-fx/view?usp=sharing). To run experiments for METABRIC dataset you need to place downloaded `BRCA1View20000.mat` file directly in the `datasets` directory.

`BRCA1View20000.mat` stores a dataset, which consists of 2106 samples with 20k attributes (gene expression) for each. There are no missing values in the data. The labels (subtypes of breast cancer) are distributed as follows:
| Subtype | 3   | 4   | 1   | 2   | 5   | 6   |
|---------|-----|-----|-----|-----|-----|-----|
| count   | 719 | 484 | 323 | 232 | 200 | 148 |

## Adding new dataset
To add a new dataset to the experiment you need to do the follwoing steps:

1. Create a loading function in `experiment/utils.py`, analogous to the `load_gemler_data_normed()`, and `load_metabric_data_normed()`.

2. Create a param grids for all desired preprocessing types, analogous to the `load_gemler_minmax_param_grid()`, `load_gemler_standard_param_grid()`, etc.

3. Add those paramgrids to the `PARAM_GRIDS` dictionary.

4. Add your dataset entry into the dictionary returned by the `get_datasets_dict()` function.

