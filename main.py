#######IMPORTS#######
import logging
import argparse
import importlib
import os
import sys
import platform

#######CUSTOM IMPORTS#######
from utilities.config_utilities import check_config_file, create_experiment_name
from utilities.data_utlities import download_datasets_and_partition_the_datasets, all_hospital_indexes_to_train_and_test_dicts, get_train_and_test_idxs_for_server, number_of_classes_from_datasets
from utilities.logger_setup import create_logger

from components.servers.clustered_sequential_server import ClusteredSequentialServer
from components.servers.normal_server import NormalServer 
from components.servers.sequential_server import SequentialServer 
from components.hospitals.fedprox_hospital import FedProxHospital
from components.hospitals.normal_hospital import NormalHospital
from torch.cuda import is_available as cuda_available
from torch.cuda import get_device_name as get_gpu_name

def select_server_and_hospital(config_dict):
    algorithm = config["algorithm"].lower() 
    if algorithm in ["fedavg", "fedprox", "fedbn"]:
        server = NormalServer
    elif algorithm == "csfl":
        server = ClusteredSequentialServer
    else:
        server = SequentialServer

    if algorithm == "fedprox":
        hospital = FedProxHospital
    else:
        hospital = NormalHospital

    return server, hospital

if __name__ == '__main__':
    platform_info = platform.uname()


    # We create the arguments parser
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", help="Path to the config file")
    parser.add_argument("-v", "--verbose", help="Flag for detailed outputs", action="store_true")
    args = parser.parse_args()

    # Check if the config argument is provided
    if args.config:
        config_file_path = args.config[:-3] if args.config.endswith('.py') else args.config
 
        config = importlib.import_module(config_file_path).config

        if check_config_file(config):
            if "experiment_name" not in config.keys():
                config["experiment_name"] = create_experiment_name(config)
        else:
            sys.exit("The config file is not valid.")

        # Set verbose based on the presence of the verbose argument
        verbose = args.verbose
        config["verbose"] = verbose
        # Use verbose variable in your code
        if verbose:
            print("Verbose output enabled.")
        else:
            print("Verbose output disabled.")
    else:
        print("Please provide the path to the config file using the -c or --config option. See README for details")
        sys.exit(1)
    
    if verbose:
        log_level = logging.DEBUG
    else:
        log_level = logging.INFO
    log_path = os.path.join(config["results_path"], f'{config["experiment_name"]}.log')
    os.makedirs(config["results_path"], exist_ok=True)
    config["log_path"] = log_path
    with open(log_path, "w") as f:
        f.write(log_path + "\n")
        f.close()
    
    # Add handlers to the logger
    logger = create_logger(log_path)
    logger.info('Initialized logger')
    if "device" in config.keys():
        if config["device"].lower() == "cpu":
            logger.info(platform_info.processor)
            device = "cpu"
        elif "cuda" in config["device"].lower():
            if cuda_available():
                logger.info("Running on GPU")
                logger.info(get_gpu_name(0))
                device = "cuda"

            else:
                logger.warning("GPU is not available")
                logger.info("Running on CPU")
                logger.info(platform_info.processor)
                device = "cpu"
    else:
        if cuda_available():
                logger.info("Running on GPU")
                logger.info(get_gpu_name(0))
                device = "cuda"
        else:
            logger.warning("GPU is not available")
            logger.info("Running on CPU")
            logger.info(platform_info.processor)
            device = "cpu"
    config["device"] = device
    # Selecting appropriate server and hospital classes
    server_type, hospital_type = select_server_and_hospital(config)
    config["node_type"] = hospital_type
    config["server_type"] = server_type
    logger.info('Selected node type %s and server type %s'%(hospital_type, server_type))


    if "model" in config.keys():
        # Selecting appropriate model
        if config["model"].lower().startswith("vgg11"):
            if config["model"].lower() == "vgg11":
                from components.models.vgg11 import VGG11
                model_type = VGG11
            else:
                raise NotImplementedError("Unknown vgg model type: %s" % config["model"])

        elif config["model"].lower().startwith("resnet"):
            match config["model"].lower():
                case "resnet18":
                    from components.models.resnet import ResNet18
                    model_type = ResNet18

                case "resnet34":
                    from components.models.resnet import ResNet34
                    model_type = ResNet34

                case "resnet50":
                    from components.models.resnet import ResNet50
                    model_type = ResNet50

                case "resnet101":
                    from components.models.resnet import ResNet152
                    model_type = ResNet152

                case "resnet152":
                    from components.models.resnet import ResNet152
                    model_type = ResNet152

                case _:
                    logger.error("Unknown resnet type %s" % config["model"])
                    raise NotImplementedError("Unknown resnet type %s" % config["model"])
        else:
            from components.models.vgg11 import VGG11
            model_type = VGG11
    config["model"] = model_type
    logger.debug('Selected model type %s'%model_type)

    config["num_classes"] = number_of_classes_from_datasets(config["datasets"])
    logger.debug('Number of classes %d'%config["num_classes"])
    hospitals_datasets, server_datasets, hospital_indices = download_datasets_and_partition_the_datasets(config)
    logger.debug("Completed data downloading/ loading and partitioning")
    datasets_names = list(config["datasets"])
    data_dict, hospitals_test_data_dict = all_hospital_indexes_to_train_and_test_dicts(hospital_indices)
    logger.debug("Completed dividing indexes to train and test for each hospital")
    server_train_idxs, server_test_idxs =  get_train_and_test_idxs_for_server([len(dataset) for dataset in server_datasets])
    logger.debug("Completed dividing indexes to train and test for the server")
    logger.info("Initialization of datasets and partitioning successful")
    logger.info("Initializing the server class")
    server = server_type(config=config, data_dicts=data_dict,
                         test_data_dicts=hospitals_test_data_dict,
                         hospitals_datasets=hospitals_datasets,
                         global_datasets=server_datasets,
                         names=list(config["datasets"]),
                         server_train_idxs=server_train_idxs,
                         server_test_idxs=server_test_idxs)
    logger.info("Starting training process")
    server.train()