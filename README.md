# Toy example

This code performs [feature selection](https://scikit-learn.org/stable/modules/feature_selection.html) and [dimensionality reduction](https://scikit-learn.org/stable/modules/unsupervised_reduction.html) with most of the sklearn methods on the [QSAR-TID-11](https://www.openml.org/search?type=data&status=active&id=46953) dataset and then runs TabPFN on the resulting data for the binary classification task. 

The original dataset consists of 5,742 samples and 1,024 features + a target variable.

To download the dataset:
```
wget https://api.openml.org/data/download/22125264/dataset
```

To run sklearn methods and TabPFN (*please adjust the run_baselines.sh and run_average.sh to your cluster specification*):
```
bash run_all.sh
```

The results are stored in the results/ directory. Please, refer to the averaged_rmse.txt for the final performances. 