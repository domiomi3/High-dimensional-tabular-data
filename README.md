# Feature selection for high dimensional tabular data

This code performs [feature selection](https://scikit-learn.org/stable/modules/feature_selection.html) and [dimensionality reduction](https://scikit-learn.org/stable/modules/unsupervised_reduction.html) with most of the available sklearn methods on the (high dimensional) OpenML datasets and runs tabular models from TabArena (TabPFNv2, CatBoost etc.) with the resulting data representations for the dataset-specified task. 
The code now also supports feature selection using [SAND layer](https://arxiv.org/abs/2505.03923) and inference with [TabPFN-Wide](https://arxiv.org/abs/2510.06162). 

## Installation

Clone the repository with the submodules (tabrepo and TabPFN-Wide):
```
git clone https://github.com/domiomi3/High-dimensional-tabular-data.git
git submodule init
git submodule update
```

Create and activate the environment:
```
uv venv
uv sync --frozen
source .venv/bin/activate
```

Install the high_tab module:
```
pip install -e .
```

### TabPFN-Wide
Installed a forked version of TabPFN-Wide so that it's compatible with TabPFN version 2.2.1.

### AutoGluon override
CatBoostModel needs a "is_train=True" argument in the internal preprocessing to be compatible with the model-specific preprocessing.
Change line 148 in autogluon.tabular.models.catboost.catboost_model.CatBoostModel to: 
```
X = self.preprocess(X, is_train=True)
```

## Usage

Create slurm script for the desired OpenML task/custom dataset, TabArena model, and FS/DR method with the following command:
```
bash experiments/generate_slurm_scripts.sh "<model(s)>" "<dataset(s)>" "<method(s)>" -g <exp-group-name> 
```

or run the training script directly:
```
python -m high_tab.train \
--openml_id task-id OR --csv_path path.csv \
--model model-name
--methods all/method-name/method-list \
--exp_group exp-name \
```

For more options, see the scripts.

## Datasets

TabArena high-dimensional datasets task IDs:
```
Bioresponse: 363620 (default)
hiva: 363677 
QSAR-TID-11: 363697 
```

### CSV data
[QSAR-oral-toxicity](https://archive.ics.uci.edu/dataset/508/qsar+oral+toxicity) - columns names added; delimiter changed from ";" to ","

The results are saved to .csv file along with config.yaml under the experiments/results/exp-group/run-id/dataset-name/model-name/method-name directory.


### Comments
Need to use TabPFN-wide fork to work with the newest tabpfn repo.
Need to pass is_train=True to CatBoost in AG (line 149) to work correctly with the override of _preprocess