import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import argparse
import os

from src.utils.plotting import load_results


def plot_val_test_over_folds(df_list, metric_name, dataset_name=None,
                             method_map=None, title=None, figsize=(20, 12), 
                             figure_dir=None, figure_name=None):
    """
    Plot validation and test performance over folds for different methods.
    Each method gets its own subplot showing test and validation scores across folds.
    
    Parameters:
    -----------
    df_list : list of DataFrames
        List of DataFrames, each containing results for one method
    metric_name : str
        Name of the metric being plotted (e.g., "roc_auc")
    dataset_name : str, optional
        Name of the dataset for the title
    method_map : dict, optional
        Mapping from method names to display names
    title : str, optional
        Custom title for the plot. If None, uses dataset_name
    figsize : tuple
        Figure size (width, height)
    figure_dir : str, optional
        Directory to save the figure
    figure_name : str, optional
        Name of the figure file (without extension)
    """
    # Default mapping if not provided
    if method_map is None:
        method_map = {}
    
    # Filter dataframes that have validation results
    valid_dfs = []
    for df in df_list:
        if f"val_{metric_name}" in df.columns:
            valid_dfs.append(df)
    
    if not valid_dfs:
        print("No validation results found in the data.")
        return
    
    # Limit to max 10 methods
    if len(valid_dfs) > 10:
        print(f"Warning: Found {len(valid_dfs)} methods, limiting to first 10")
        valid_dfs = valid_dfs[:10]
    
    n_methods = len(valid_dfs)
    
    # Calculate grid dimensions (prefer wide layout)
    n_cols = min(3, n_methods)
    n_rows = (n_methods + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize, squeeze=False)
    axes = axes.flatten()
    
    # Determine if lower is better or higher is better
    lower_is_better = metric_name.lower() in ['log_loss', 'rmse', 'mse', 'mae', 'error']
    
    # Sort methods by mean test performance
    method_performance = []
    for df in valid_dfs:
        method_name = df["method"].iloc[0]
        mean_perf = df[metric_name].mean()
        method_performance.append((method_name, mean_perf, df))
    
    if lower_is_better:
        method_performance.sort(key=lambda x: x[1])
    else:
        method_performance.sort(key=lambda x: x[1], reverse=True)
    
    # Get dataset name from first dataframe if not provided
    if dataset_name is None and len(valid_dfs) > 0:
        dataset_name = valid_dfs[0]["dataset_name"].iloc[0]
    
    # Plot each method
    for idx, (method_name, _, df) in enumerate(method_performance):
        ax = axes[idx]
        
        # Sort dataframe by repeat and fold for consistent ordering
        df_sorted = df.sort_values(['repeat', 'fold']).reset_index(drop=True)
        
        # Create x-axis labels (repeat_fold format)
        x_labels = [f"{int(r)}x{int(f)}" for r, f in zip(df_sorted['repeat'], df_sorted['fold'])]
        x_pos = np.arange(len(df_sorted))
        
        # Plot test and validation scores
        test_scores = df_sorted[metric_name].values
        val_scores = df_sorted[f"val_{metric_name}"].values
        
        ax.plot(x_pos, test_scores, marker='o', linestyle='-', 
                linewidth=2, markersize=6, label='test', color='steelblue', alpha=0.8)
        ax.plot(x_pos, val_scores, marker='s', linestyle='-', 
                linewidth=2, markersize=6, label='val', color='coral', alpha=0.8)
        
        # Add mean lines
        test_mean = test_scores.mean()
        val_mean = val_scores.mean()
        ax.axhline(y=test_mean, color='steelblue', linestyle='--', 
                   linewidth=1.5, alpha=0.5, label=f'test mean: {test_mean:.3f}')
        ax.axhline(y=val_mean, color='coral', linestyle='--', 
                   linewidth=1.5, alpha=0.5, label=f'val mean: {val_mean:.3f}')
        
        # Styling
        display_name = method_map.get(method_name, method_name)
        ax.set_title(display_name, fontsize=12, fontweight='bold')
        ax.set_xlabel('repeatxfold', fontsize=10)
        ax.set_ylabel(metric_name.replace('_', ' ').lower(), fontsize=10)
        ax.grid(axis='y', alpha=0.3)
        ax.legend(fontsize=8, loc='best')
        
        # Set x-ticks
        if len(x_labels) <= 20:
            ax.set_xticks(x_pos)
            ax.set_xticklabels(x_labels, rotation=45, ha='right', fontsize=8)
        else:
            # Show fewer labels if too many
            step = max(1, len(x_labels) // 10)
            ax.set_xticks(x_pos[::step])
            ax.set_xticklabels([x_labels[i] for i in range(0, len(x_labels), step)], 
                              rotation=45, ha='right', fontsize=8)
    
    # Hide unused subplots
    for idx in range(n_methods, len(axes)):
        axes[idx].axis('off')
    
    # Overall title
    if title:
        fig.suptitle(title, fontsize=18, fontweight='bold', y=0.995)
    elif dataset_name:
        fig.suptitle(f'{dataset_name} - Test vs Validation Performance', 
                    fontsize=18, fontweight='bold', y=0.995)
    
    plt.tight_layout()
    
    # Save figure if directory and name provided
    if figure_dir and figure_name:
        os.makedirs(figure_dir, exist_ok=True)
        save_path = os.path.join(figure_dir, f"{figure_name}.png")
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to: {save_path}")
    
    plt.show()


def main():
    args = parser.parse_args()
    
    # Load results from the root directory
    df_list, metric = load_results(args.root_dir)
    
    if not df_list:
        print("No results found in the provided directory.")
        return
    
    # Get dataset name from first dataframe
    dataset = df_list[0]["dataset_name"].iloc[0] if len(df_list) > 0 else None
    
    # Optional: Define method display names
    method_map = {
        'random_fs': 'random fs',
        'variance_fs': 'variance fs',
        'tree_fs': 'tree-based fs',
        'kbest_fs': 'k-best fs',
        'pca_dr': 'pca dr',
        'random_dr': 'random projection dr',
        'kpca_dr': 'kernel pca dr',
        'agglo_dr': 'agglomerative dr',
        'kbest+pca': 'k-best+pca',
        'tabpfnv2_tab': 'TabPFNv2'
    }
    
    plot_val_test_over_folds(
        df_list=df_list,
        metric_name=metric,
        dataset_name=dataset,
        method_map=method_map,
        title=args.title,
        figsize=(args.fig_width, args.fig_height),
        figure_dir=args.figure_dir,
        figure_name=args.figure_name
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Plot validation and test performance over folds for multiple methods"
    )
    parser.add_argument("--root_dir", required=True, 
                       help="Root directory containing CSV results")
    parser.add_argument("--title", default=None, 
                       help="Custom title for the plot")
    parser.add_argument("--fig_width", type=float, default=20, 
                       help="Figure width (default: 20)")
    parser.add_argument("--fig_height", type=float, default=12, 
                       help="Figure height (default: 12)")
    parser.add_argument("--figure_dir", default="experiments/results/figures/val_test_folds", 
                       help="Directory to save figures")
    parser.add_argument("--figure_name", default="val_test_folds", 
                       help="Saved as .png")
    
    main()

    # python -m experiments.plot_val --root_dirs "/work/dlclarge2/matusd-toy_example/experiments/results/tabarena_map/gpu_complete/qsar_tid_11/tabpfnv2_tab/" --title "Feature Selection Comparison" --figure_dir experiments/figures/val_vs_test --figure_name qsar_tabpfnv2_val_test