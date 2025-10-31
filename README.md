# FairRecAssessment: Fair Assessment of Fair Recommender Systems: An Experimental Framework on Model and Data Characteristics

FairRecAssessment is a reproducible and extensible benchmarking toolkit built upon [RecBole](https://recbole.io), designed to evaluate and compare fairness-aware recommendation models under a unified framework.

## Highlights

- **Unified evaluation**: Multiple accuracy and fairness metrics across different datasets and sensitive attributes.
- **Extensible design**: New models and metrics can be easily integrated using RecBole’s modular interface.
- **Comprehensive comparisons**: Supports both vanilla and fairness-enhanced recommender models for side-by-side analysis.
- **Dataset-aware analysis**: Includes tools to analyze how data characteristics (e.g., imbalance, popularity, density) affect fairness–accuracy trade-offs.


## Requirements
You can download the requirements by using the requirements.txt file.
```
git clone https://github.com/ITU-ILAP/Fair-Recommender-Assessment-Tool.git
cd Fair-Recommender-Assessment-Tool
pip install -r requirements.txt
```

## Quick-Start

With the source code, you can use the provided script to run multiple recommender models under the same setup:

```
python new_run_compare.py
```
If you want to run the regression analysis using the recommender model results to understand the significance of the features:
```
python ./evaluation/regression.py
```
## Detailed Implementations

1. **calculate_stats.py**

This script computes comprehensive statistical summaries for recommendation datasets used in fairness-aware recommender system research.
It analyzes multiple data subsets and extracts key user–item interaction and fairness-related statistics.
Usage
```
python calculate_statst.py
```
You can modify the parameters inside the script:
```
base_path = "dataset_v2/ml-1M"
dataset_name = "ml1m"
user_file = "dataset_v2/ml-1M/ml-1M.user"
sensitive_col = "gender:float"
output_file = "stats/stats_ml1m_gender.csv"
subsets = ["URM_subsets_filtered"]
```

Running the script generates a file such as:
```
stats/stats_ml1m_gender.csv
```

2. **build_df_regression.ipynb** (dataset builder)

This notebook merges model evaluation results (fairness and accuracy metrics) with data characteristics (from the stats files) to produce the df_regression.csv file, which serves as the input for regression.py.

3. **regression.py**

This script runs the regression analyses used in the paper’s RQ1–RQ2:
- RQ1 (Fairness-focused): Explains which data characteristics (and group indicators) are associated with each fairness metric.
- RQ2 (Accuracy-focused): Explains which data characteristics are associated with accuracy metrics.
It supports model-based slicing (per model) or pooled analysis across all models.
Usage
```
python regression.py
```

4. **regression_analysis.ipynb**

This notebook analyzes the significance and effect directions of features produced from df_regression.csv under different scenarios (per-model, pooled, per sensitive feature). It complements the OLS outputs by ranking influential variables and visualizing fairness–accuracy relationships.

🔍 What it does

- Loads df_regression.csv (pre-built panel with subsets × models × metrics × data characteristics).

- Drops selected fairness metrics and data characteristics (configurable lists).

- If model_based=True, iterates over model_list = ["NFCF", "FOCF", "PFCN_MLP"] and:

  - Runs OLS for each fairness metric (RQ1).

  - Runs OLS for each accuracy metric (RQ2).

  - Prints OLS summaries and concatenates results.

- If model_based=False, runs the same analyses on the pooled data.

Configuration (inside the script)

- model_list = ["NFCF", "FOCF", "PFCN_MLP"]

- model_based = True → per-model analiz 

- dropped_fairness_measures, dropped_dc_08, dropped_dc_07 → which metrics/data characteristics will be excluded
  
## Implement Models

We list the models that we have implemented up to now:

- [FOCF](recbole/model/fair_recommender/focf.py) from Sirui Yao et al:[Beyond Parity：Fairness Objectives for Collaborative Filtering](https://proceedings.neurips.cc/paper/2017/hash/e6384711491713d29bc63fc5eeb5ba4f-Abstract.html)(NIPS 2017).
- PFCN from Yunqi Li et al:[Towards Personalized Fairness based on Causal Notion](https://dl.acm.org/doi/abs/10.1145/3404835.3462966?casa_token=zzHePKuKP6AAAAAA:YzZp_qUbzsgd3TXWCAGSRAfEHO2oM0_BuWZ5uZlfj_rudqKGYq8douOaZ0GoizxP54jtz3JDFw725xo)(SIGIR 2021)
  - [PFCN_MLP](recbole/model/fair_recommender/pfcn_mlp.py)
  - [PFCN_BiasedMF](recbole/model/fair_recommender/pfcn_biasedmf.py)
  - [PFCN_DMF](recbole/model/fair_recommender/pfcn_dmf.py)
  - [PFCN_PMF](recbole/model/fair_recommender/pfcn_pmf.py)
- [NFCF](recbole/model/fair_recommender/nfcf.py) from Rashidul Islam et al:[Debiasing career recommendations with neural fair collaborative filtering](https://dl.acm.org/doi/abs/10.1145/3442381.3449904?casa_token=ZzbZbC-Fn_oAAAAA:6KCSThLs7UsT9s0ZzeSryT3Mry067KeTiNdurfa9Q9UHWY7fLGgmjPtQy9i1zU1Yqm4Xf46NVYVuu40) (WWW 2021) 

## Datasets

 The datasets used can be downloaded from [GroupLens](https://grouplens.org/datasets/movielens/1m/).

# Hyper-parameters
We train the models with the default parameter settings, suggested in their original paper.[[link]](results/ml-1m.md)

## Acknowledgement

The implementation is based on the open-source recommendation library [RecBole](https://github.com/RUCAIBox/RecBole) and [FairRec](https://github.com/TangJiakai/RecBole-FairRec#) .
