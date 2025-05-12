import re
import os 

from collections import defaultdict

def average_rmse(seed_list):
    result_files = [f"results/results_seed_{seed}.txt" for seed in seed_list]
    method_rmse = defaultdict(list)

    for filename in result_files:
        try:
            with open(filename, "r") as file:
                content = file.read()
        except FileNotFoundError:
            print(f"Warning: {filename} not found. Skipping.")
            continue

        chunks = content.strip().split('----------------------------------------')
        for chunk in chunks:
            if not chunk.strip():
                continue
            name_match = re.search(r"NAME: (.+)", chunk)
            rmse_match = re.search(r"RMSE: ([\d\.]+)", chunk)
            if name_match and rmse_match:
                method = name_match.group(1).strip()
                rmse = float(rmse_match.group(1))
                method_rmse[method].append(rmse)

    os.makedirs("results", exist_ok=True)
    with open("results/averaged_rmse.txt", "w") as f:
        f.write("Average RMSE per method across seeds:\n")
        for method, rmses in method_rmse.items():
            avg_rmse = sum(rmses) / len(rmses)
            f.write(f"{method:12s}: {avg_rmse:.6f} (n={len(rmses)})\n")

def main():
    with open("used_seeds.txt", "r") as f:
        seeds = f.read().strip().split()

    average_rmse(seeds)

if __name__ == "__main__":
    main()
