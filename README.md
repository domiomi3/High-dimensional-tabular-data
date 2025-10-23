# Learning representations of high dimensional tabular data

This code performs [feature selection](https://scikit-learn.org/stable/modules/feature_selection.html) and [dimensionality reduction](https://scikit-learn.org/stable/modules/unsupervised_reduction.html) with most of the available sklearn methods on the (high dimensional) OpenML datasets and runs tabular models from TabArena (TabPFNv2, CatBoost etc.) with the resulting data representations for the dataset-specified task. 


Clone the repository with the submodules (tabrepo and TabPFN-Wide):
```
git clone --recurse-submodules fsdjk
```
or if already cloned:
```
git submodule update --init --recursive
```

Create and activate the environment:
```
uv venv
uv sync --frozen
source .venv/bin/activate
```

Create slurm script for the desired OpenML task/dataset, TabArena model, and sklearn method and submit the job:
```
python experiments/generate_slurm_script.py \
--openml_id task-id/dataset-name OR --csv_path path.csv \
--model model-name
--methods all/method-name/method-list  \
--exp_group exp-name \
```

or run the training script directly:
```
python src/train.py \
--openml_id task-id OR --csv_path path.csv \
--model model-name
--methods all/method-name/method-list \
--exp_group exp-name \
```

**exp-name** is directory created under experiments/results/ 

TabArena high-dimensional datasets task IDs:
```
Bioresponse: 363620 (default)
hiva: 363677 
QSAR-TID-11: 363697 
```

The results are saved to .csv file along with config.yaml under the experiments/results/dataset-name/model-name/method-name directory.


# TabPFN-Wide
Installed a forked version of TabPFN-Wide so that it's compatible with TabPFN version 2.2.1.

# TabRepo
Installed a forked version of TabRepo so the internal preprocessing works (added is_train=True).