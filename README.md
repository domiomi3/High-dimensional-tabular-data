# Toy example

This code performs [feature selection](https://scikit-learn.org/stable/modules/feature_selection.html) and [dimensionality reduction](https://scikit-learn.org/stable/modules/unsupervised_reduction.html) with most of the sklearn methods on the [QSAR-TID-11](https://www.openml.org/search?type=data&status=active&id=46953) dataset and then runs TabPFN on the resulting data for the binary classification task. 

The original dataset consists of 5,742 samples and 1,024 features + a target variable.

Install conda and create environment:
```
conda env create -f env.yaml
```

To run sklearn methods and TabPFN (or pass any subset of the methods):
```
sklearn run_baselines.py --method original random_fs variance_fs tree_fs pca_dr random_dr agglo_dr 
```
or
```
bash run_baselines.sh
```

The results are saved to .csv files in the results/ directory.