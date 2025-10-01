import os
import pandas as pd

def load_results(rootdir, requested_columns=[]):
    columns = ["dataset_name", "method", "model", "repeat", "fold"] + requested_columns
    metrics=["log_loss", "roc_auc", "rmse"]
    df_list = []

    for subdir, _, files in os.walk(rootdir):
        if any([filename.endswith('.csv') for filename in files]):
            for filename in files:
                if filename.endswith('.csv'):
                    csv_path = os.path.join(subdir, filename)
                    df_all = pd.read_csv(csv_path)
                    metric = [m_col for m_col in df_all.columns if m_col in metrics][0]
                    val_metric = [m_col for m_col in df_all.columns if m_col == f"val_{metric}"][0]
                    columns_req = columns + [metric] + [val_metric]
                    df_req = df_all[columns_req].copy()
                    df_list.append(df_req)

    return df_list, metric


def get_mean_std(df_list, metric, if_val=True):
    perf_dict = {}
    for df_single in df_list:
        method = df_single["method"][0]            
        perf_dict[method] = {}
        perf_dict[method]["mean"] = df_single[metric].mean()
        perf_dict[method]["std"] = df_single[metric].std()
        
        if f"val_{metric}" in df_single.columns and if_val:
            perf_dict[method]["val_mean"] = df_single[f"val_{metric}"].mean()
            perf_dict[method]["val_std"] = df_single[f"val_{metric}"].std()
    return perf_dict


def merge_results(root_dirs):
    comb_results = {}
    for root_dir in root_dirs:
        results, metric = load_results(rootdir=root_dir)
        model = results[0]["model"][0]
        dataset = results[0]["dataset_name"][0]
        avg_results = get_mean_std(results, metric)
        comb_results[model] = avg_results
    
    return comb_results, dataset, metric
