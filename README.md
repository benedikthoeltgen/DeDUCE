# DeDUCE

## Prelims: model training
`run_training.py` trains ResNets, drawing on files in `DDU`.
Sanity checks for the DDU models are performed in `nb_DDU_FashionMNIST.ipynb`.

## Generating Counterfactuals
`run_DeDUCE.py`, `run_JSMA.py`, and `run_REVISE.py` generate counterfactuals, drawing on files in `CE`.

## AnoGAN Metric
`nb_AnoGAN_eval.ipynb` is used for tuning the AnoGAN metric as well as computing scores, drawing on files in `metrics`.
Sanity checks are performed in `nb_metrics_EMNIST.ipynb`.

## Tuning the Algortihms
`nb_tune_DeDUCE.ipynb` and `nb_tune_REVISE.ipynb` are used for tuning the respective algorithm on the validation set examples in `valset_batch`.

## Visualising Results
`nb_eval_testset.ipynb` performs the testset evalutation, with files in `_testset_results`.
`nb_eval_testset2-5.ipynb` performs further evalutations, on additional testsets, with files in `_testset_results*`.
