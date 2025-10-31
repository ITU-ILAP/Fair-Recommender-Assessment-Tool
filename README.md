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
python>=3.7.0
recbole>=1.0.1
numpy>=1.20.3
torch>=1.11.0
tqdm>=4.62.3
```

## Quick-Start

With the source code, you can use the provided script to run multiple recommender models under the same setup:

```
python run_compare.py
```
If you want to run the regression analysis using the recommender model results to understand the significance of the features:
```
python ./evaluation/regression.py
```

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
