python results_summary.py results_2023_05_23_1732_PCA_WHOLE_DATA --score_groups individual --skip_table --scaler standard
python results_summary.py results_2023_05_24_1020_StandardScaler_WHOLE_DATA --score_groups individual --skip_table --scaler standard
python results_summary.py results_2023_05_24_1508_QuantileTransformer_WHOLE_DATA --score_groups individual --skip_table --scaler quantile
python results_summary.py results_2023_05_24_2340_MinMaxScaler_WHOLE_DATA --score_groups individual --skip_table --scaler min-max
# python results_summary.py \
#     results_2023_05_23_1732_PCA_WHOLE_DATA,results_2023_05_24_1020_StandardScaler_WHOLE_DATA,results_2023_05_24_1508_QuantileTransformer_WHOLE_DATA,results_2023_05_24_2340_MinMaxScaler_WHOLE_DATA \
#     --score_groups individual --skip_table