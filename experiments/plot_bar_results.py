import matplotlib.pyplot as plt
import numpy as np
import argparse
import os
import pandas as pd

from src.utils.plotting import merge_results

DISPLAY_NAME_MAP = {
    "tabpfnv2_tab_map": "TabPFNv2 (MAP)",
    "tabpfnv2_tab": "TabPFNv2 (MSP)",
    # "tabpfnv2_org": "TabPFNv2 Original",
    "catboost_tab": "CatBoost (1 GPU, 1 CPU)",
}

CAT_DISPLAY_NAME_MAP = {
    "catboost_tab": "CatBoost (1 GPU, 1 CPU)",
    "catboost_tab_cpu": "CatBoost (8 CPU)",
}

COLOR_MAP = {
    "tabpfnv2_tab": "#a6d8af",
    "tabpfnv2_tab_map": "#f4d8af",
    "catboost_tab": "#afc8f4",
}

CAT_COLOR_MAP = {
    "catboost_tab_cpu": "#f4d8af",
    "catboost_tab": "#ac569e"
}

def plot_test_performance(results_dicts, metric_name, dataset_name=None, 
                            model_map=None, color_map=None, sota_models=None, title=None,
                            baseline="tabpfnv2_tab", figsize=(14, 6), y_margin=0.1,
                            figure_dir="", figure_name="test.png", save_summary=True,
                            summary_dir=""):
    """
    Plot performance metrics from multiple sources.
    
    Parameters:
    -----------
    results_dicts : dict of dict
        Dictionary where keys are source names (e.g., "tab1", "tab2") 
        and values are performance dictionaries from get_mean_std()
    metric_name : str
        Name of the metric being plotted (e.g., "log_loss")
    dataset_name : str, optional
        Name of the dataset for the title. If None, uses "Performance Comparison"
    model_map : dict, optional
        Mapping from model names to display names (e.g., {"tab1": "Tabular Model 1"})
    color_map : dict, optional
        Mapping from model names to colors (e.g., {"tab1": "#FF6B6B"})
    sota_models : dict, optional
        Dictionary with SOTA model results, format: {"model_name": {"mean": val, "std": val}}
    baseline : str
        Model name to use as baseline for ordering methods (default: "tabpfnv2_tab")
    figsize : tuple
        Figure size (width, height)
    y_margin : float
        Absolute margin to add above and below min/max values (default: 0.1)
    """
    # create output dir
    os.makedirs(figure_dir, exist_ok=True)
    os.makedirs(summary_dir, exist_ok=True)

    figure_path = os.path.join(figure_dir, figure_name+".png")

    # Default mappings if not provided
    if model_map is None:
        model_map = {k: k for k in results_dicts.keys()}
    if color_map is None:
        color_map = {}
    
    # Get all unique methods across all sources
    all_methods = set()
    for perf_dict in results_dicts.values():
        all_methods.update(perf_dict.keys())
    
    # Determine if lower is better or higher is better
    lower_is_better = metric_name.lower() in ['log_loss', 'rmse', 'mse', 'mae', 'error']
    
    # Order methods by baseline performance
    if baseline in results_dicts and results_dicts[baseline]:
        # Sort by baseline model's mean performance
        baseline_dict = results_dicts[baseline]
        if lower_is_better:
            methods = sorted(all_methods, 
                            key=lambda m: baseline_dict[m]["mean"] if m in baseline_dict else float('inf'))
        else:
            methods = sorted(all_methods, 
                            key=lambda m: baseline_dict[m]["mean"] if m in baseline_dict else float('-inf'),
                            reverse=True)
    else:
        methods = sorted(list(all_methods))
    
    # # Add SOTA models to the appropriate end (where best results are)
    if sota_models:
        sota_method_names = list(sota_models.keys())
    methods = sota_method_names + methods
    
    # Prepare data
    sources = list(results_dicts.keys())
    n_sources = len(sources)
    n_methods = len(methods)
    
    fig, ax = plt.subplots(figsize=figsize)
    x_pos = np.arange(n_methods)
    width = 0.8 / n_sources  # Divide available space by number of sources
    
    # Collect all values for y-axis range calculation (including error bars)
    all_lower_bounds = []
    all_upper_bounds = []
    
    # Plot bars for each source
    for i, source in enumerate(sources):
        perf_dict = results_dicts[source]
        means = []
        stds = []
        
        if save_summary:
            csv_filename = f"{dataset_name}_{source}.csv"
            csv_path = os.path.join(summary_dir, csv_filename)
            df = pd.DataFrame(perf_dict).T
            df.index.name = 'method'
            df.to_csv(csv_path)

        for method in methods:
            if method in perf_dict:
                mean_val = perf_dict[method]["mean"]
                std_val = perf_dict[method]["std"]
                means.append(mean_val)
                stds.append(std_val)
                all_lower_bounds.append(mean_val - std_val)
                all_upper_bounds.append(mean_val + std_val)
            else:
                means.append(0)
                stds.append(0)
        
        offset = (i - n_sources/2 + 0.5) * width
        display_name = model_map.get(source, source)
        color = color_map.get(source, None)
        
        ax.bar(x_pos + offset, means, width, yerr=stds, 
               capsize=3, alpha=0.7, label=display_name, 
               color=color, ecolor='black')
    
    # Plot SOTA models if provided
    if sota_models:
        sota_means = []
        sota_stds = []
        
        for method in methods:
            if method in sota_models:
                mean_val = sota_models[method]["mean"]
                std_val = sota_models[method]["std"]
                sota_means.append(mean_val)
                sota_stds.append(std_val)
                all_lower_bounds.append(mean_val - std_val)
                all_upper_bounds.append(mean_val + std_val)
            else:
                sota_means.append(np.nan)
                sota_stds.append(0)
        
        # Filter out NaN values for plotting
        valid_indices = [i for i, v in enumerate(sota_means) if not np.isnan(v)]
        valid_x_pos = [x_pos[i] for i in valid_indices]
        valid_means = [sota_means[i] for i in valid_indices]
        valid_stds = [sota_stds[i] for i in valid_indices]
        
        if valid_means:
            ax.bar(valid_x_pos, valid_means, width * n_sources, yerr=valid_stds,
                   capsize=3, alpha=0.7, label='TabArena SOTA', color='purple', ecolor='black')
    
    if "original" in methods:
        original_idx = methods.index("original")
        # Get the mean value for "original" from the baseline model
        if baseline in results_dicts and "original" in results_dicts[baseline]:
            original_mean = results_dicts[baseline]["original"]["mean"]
            
            # Draw horizontal line across the plot
            ax.axhline(y=original_mean, color='red', linestyle='--', 
                    linewidth=1, alpha=0.8)
            
    # Set y-axis range with absolute margin
    if all_lower_bounds and all_upper_bounds:
        y_min = min(all_lower_bounds)
        y_max = max(all_upper_bounds)
        ax.set_ylim(y_min - y_margin, y_max + y_margin)
    
    ax.set_xlabel('FS/DR method', fontsize=12)
    ax.set_ylabel(metric_name.replace('_', ' ').lower(), fontsize=12)
    
    # Set title
    if dataset_name:
        title = title
    else:
        title = f'{dataset_name}'
    ax.set_title(title, fontsize=14)
    
    ax.set_xticks(x_pos)
    ax.set_xticklabels(methods, rotation=45, ha='right')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()

    plt.savefig(figure_path)


def main():
    args = parser.parse_args()
    
    # load sota models
    sota_models = None
    if args.sota_csv:
        import pandas as pd
        sota_df = pd.read_csv(args.sota_csv)
        sota_models = {}
        for _, row in sota_df.iterrows():
            sota_models[row['model']] = {
                'mean': row['mean'],
                'std': row['std']
            }
    
    results_dicts, dataset, metric = merge_results(args.root_dirs)

    plot_test_performance(
        results_dicts=results_dicts, 
        metric_name=metric, 
        dataset_name=dataset, 
        model_map=CAT_DISPLAY_NAME_MAP, 
        color_map=CAT_COLOR_MAP, 
        sota_models=sota_models,
        title=args.plot_title if args.plot_title else None,
        baseline=args.baseline, 
        figsize=(14, 6), 
        y_margin=float(args.y_margin),
        figure_dir=args.figure_dir,
        figure_name=args.figure_name,
        save_summary=args.save_summary,
        summary_dir=args.summary_dir
    )
    
if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root_dirs", nargs='+', required=True, help="List of root directories")
    parser.add_argument("--sota_csv", default=None, help="Path to CSV file with SOTA results")
    parser.add_argument("--figure_dir", default="experiments/figures/test", help="Directory to save figures")
    parser.add_argument("--figure_name", default="test", help="Saved as .png")
    parser.add_argument("--plot_title", type=str)
    parser.add_argument("--baseline", type=str)
    parser.add_argument("--save_summary", action="store_true")
    parser.add_argument("--summary_dir", default="experiments/summary/test", help="Directory to save performance summaries")
    parser.add_argument("--y_margin", type=float, default=0.1)
    main()

# python -m experiments.plot_bar_results --root_dirs /work/dlclarge2/matusd-toy_example/experiments/results/tabarena_map/cpu_catboost_complete/bioresponse/catboost_tab_cpu /work/dlclarge2/matusd-toy_example/experiments/results/tabarena_map/gpu_complete/bioresponse/catboost_tab --sota_csv /work/dlclarge2/matusd-toy_example/experiments/results/tabarena_map/gpu_complete/bioresponse/sota.csv --figure_name "bioresponse_catboost_cpu_gpu" --plot_title "Bioresponse (↑)" --baseline "catboost_tab_cpu" --save_summary --y_margin 0.01
# python -m experiments.plot_bar_results --root_dirs /work/dlclarge2/matusd-toy_example/experiments/results/tabarena_map/cpu_catboost_complete/qsar_tid_11/catboost_tab_cpu /work/dlclarge2/matusd-toy_example/experiments/results/tabarena_map/gpu_complete/qsar_tid_11/catboost_tab --sota_csv /work/dlclarge2/matusd-toy_example/experiments/results/tabarena_map/gpu_complete/qsar_tid_11/sota.csv --figure_name "qsar_catboost_cpu_gpu" --plot_title "QSAR-TID-11 (↓)" --baseline "catboost_tab_cpu" --save_summary --y_margin 0.05
# python -m experiments.plot_bar_results --root_dirs /work/dlclarge2/matusd-toy_example/experiments/results/tabarena_map/cpu_catboost_complete/hiva_agnostic/catboost_tab_cpu /work/dlclarge2/matusd-toy_example/experiments/results/tabarena_map/gpu_complete/hiva_agnostic/catboost_tab --sota_csv /work/dlclarge2/matusd-toy_example/experiments/results/tabarena_map/gpu_complete/hiva_agnostic/sota.csv --figure_name "qsar_catboost_cpu_gpu" --plot_title "hiva_agnostic (↓)" --baseline "catboost_tab_cpu" --save_summary --y_margin 0.05



# python -m experiments.plot_bar_results --root_dirs /work/dlclarge2/matusd-toy_example/experiments/results/tabarena_msp/complete/hiva_agnostic/tabpfnv2_tab /work/dlclarge2/matusd-toy_example/experiments/results/tabarena_map/gpu_complete/hiva_agnostic/tabpfnv2_tab /work/dlclarge2/matusd-toy_example/experiments/results/tabarena_map/gpu_complete/hiva_agnostic/catboost_tab --sota_csv /work/dlclarge2/matusd-toy_example/experiments/results/tabarena_map/gpu_complete/hiva_agnostic/sota.csv --figure_name "hiva_tabpfn_msp_map_catboost" --plot_title "hiva_agnostic (↓)" --baseline "tabpfnv2_tab" --save_summary --y_margin 0.01
# python -m experiments.plot_bar_results --root_dirs /work/dlclarge2/matusd-toy_example/experiments/results/tabarena_msp/complete/qsar_tid_11/tabpfnv2_tab /work/dlclarge2/matusd-toy_example/experiments/results/tabarena_map/gpu_complete/qsar_tid_11/tabpfnv2_tab /work/dlclarge2/matusd-toy_example/experiments/results/tabarena_map/gpu_complete/qsar_tid_11/catboost_tab --sota_csv /work/dlclarge2/matusd-toy_example/experiments/results/tabarena_map/gpu_complete/qsar_tid_11/sota.csv --figure_name "qsar_tabpfn_msp_map_catboost" --plot_title "QSAR-TID-11 (↓)" --baseline "tabpfnv2_tab" --save_summary --y_margin 0.05
# python -m experiments.plot_bar_results --root_dirs /work/dlclarge2/matusd-toy_example/experiments/results/tabarena_msp/complete/bioresponse/tabpfnv2_tab /work/dlclarge2/matusd-toy_example/experiments/results/tabarena_map/gpu_complete/bioresponse/tabpfnv2_tab /work/dlclarge2/matusd-toy_example/experiments/results/tabarena_map/gpu_complete/bioresponse/catboost_tab --sota_csv /work/dlclarge2/matusd-toy_example/experiments/results/tabarena_map/gpu_complete/bioresponse/sota.csv --figure_name "bioresponse_tabpfn_msp_map_catboost" --plot_title "Bioresponse (↑)" --baseline "tabpfnv2_tab" --save_summary --y_margin 0.05