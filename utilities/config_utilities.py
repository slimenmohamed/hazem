# * Algorithm
# * Datasets
# * Datasets size
# * Number of nodes (hospitals)
# * Batch size
# * Learning rate
# * Local epochs
# * Global epochs (If needed)
# * Data directory
# * Results directory
# * Verbose logs or not
# * Participation percentage
# * Mu (If needed)
# * IID /  non-IID partition

import logging
import sys

def check_config_file(config):
    necessary_items = ["algorithm", "datasets", "size_per_dataset",
                         "batch_size", "learning_rate", "local_epochs"]
    # TODO: add "number_of_hospitals" later
    missing = []
    for key in necessary_items:
        if key not in config.keys():
            missing.append(key)
    specific_missing = []
    if "algorithm" in missing:
        algorithm = config["algorithm"]
        if algorithm not in ["fedavg", "fedprox", "fedbn", "csfl", "sfl"]:
            logging.critical(f"Unknown algorithm: {algorithm}")
        if algorithm == "fedprox" and "mu" not in config.keys():
            specific_missing.append("mu")

    if len(missing) > 0:
        logging.warning(f"Missing required configuration: {missing}")
    if len(specific_missing) > 0:
        logging.warning(f"Missing required configuration for the {algorithm}: {specific_missing}")
    return len([*missing, *specific_missing]) == 0

def create_experiment_name(config) -> str:
    experiment_name = f"{config['algorithm']}_{len(list(config['datasets']))}ds_size{config['size_per_dataset']}_B{config['batch_size']}_lr{config['learning_rate']}_E{config['local_epochs']}"
    if "mu" in config.keys():
        experiment_name += f"_mu{config['mu']}"
    return experiment_name