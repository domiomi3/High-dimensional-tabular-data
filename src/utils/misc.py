def add_method_args(config):
    """
    Add relevant HPs to a dictionary passed to TabArenaModel with model-specific preprocessing.
    """
    method_args = {}
    method_args["method"] = config["method"]
    method_hps = {
        "num_features": config["num_features"], 
        "random_state": config["seed"],
        "task_type": config["task_type"]
    }
    if method_args["method"] == "kbest+pca":
        method_hps["num_kbest_features"] = config["num_kbest_features"]
        method_hps["num_pca_comps"] = config["num_pca_comps"]
    if method_args["method"] == "tree_fs":
        method_hps["num_tree_estimators" ] = config["num_tree_estimators"]
    if method_args["method"] == "variance_fs":
        method_hps["var_threshold"] = config["var_threshold"]
    method_args.update(method_hps)
    return {"method_args": method_args}