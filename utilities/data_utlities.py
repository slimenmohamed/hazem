import os
import random
import medmnist
from medmnist import INFO
import torchvision.transforms.functional as TF
import torchvision.transforms as transforms
# import logging
import numpy as np
import warnings
import sys
from utilities.logger_setup import create_logger

warnings.filterwarnings("ignore")


# fixing the seed as one value (defaults to 1, if specified in the configuration file, the seed gets updated to that value)
seed = 1
# Set the random seed for numpy
np.random.seed(seed)
# Set the random seed for Python's built-in random module
random.seed(seed)
def non_iid_partition(dataset, num_hospitals):
    """
    non I.I.D parititioning of data over hospitals
    Sort the data by the digit label
    Divide the data into N shards of size S
    Each of the hospitals will get X shards

    params:
      - dataset : Dataset, an iterable object containing pairs of (images, labels)
      - num_hospitals (int): Number of hospitals to split the data between

    returns:
      - Dictionary of image indexes for each hospital
    """

    shards_size = 9
    total_shards = len(dataset) // shards_size
    num_shards_per_hospital = total_shards // num_hospitals
    shard_idxs = [i for i in range(total_shards)]
    hospital_dict = {i: np.array([], dtype='int64') for i in range(num_hospitals)}
    idxs = np.arange(len(dataset))
    
    # get labels as a numpy array
    data_labels = np.array([np.array(target).flatten()
                           for _, target in dataset]).flatten()
    # sort the labels
    label_idxs = np.vstack((idxs, data_labels))
    label_idxs = label_idxs[:, label_idxs[1, :].argsort()]
    idxs = label_idxs[0, :]

    # divide the data into total_shards of size shards_size
    # assign num_shards_per_hospital to each hospital
    for i in range(num_hospitals):
        rand_set = set(np.random.choice(
            shard_idxs, num_shards_per_hospital, replace=False))
        shard_idxs = list(set(shard_idxs) - rand_set)

        for rand in rand_set:
            hospital_dict[i] = np.concatenate(
                (hospital_dict[i], idxs[rand*shards_size:(rand+1)*shards_size]), axis=0)
    return hospital_dict  # hospital dict has [idx: list(datapoint indices)


def divide_indices_into_test_and_train(data_indices, batch_number, num_batches):
    """
    Divide a list of data indices into training and testing sets based on the batch number and number of batches.

    Parameters:
        data_indices (list): List of data indices.
        batch_number (int): The batch number.
        num_batches (int): The total number of batches.

    Returns:
        tuple: A tuple containing the training data indices and testing data indices.

    Example:
        data_indices = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        num_batches = 3

        # Batch 0
        train_data, test_data = divide_indices_into_test_and_train(data_indices, 0, num_batches)
        print(train_data)  # Output: [4, 5, 6, 7, 8, 9, 10]
        print(test_data)   # Output: [1, 2, 3]

        # Batch 1
        train_data, test_data = divide_indices_into_test_and_train(data_indices, 1, num_batches)
        print(train_data)  # Output: [1, 2, 3, 7, 8, 9, 10]
        print(test_data)   # Output: [4, 5, 6]

        # Batch 2
        train_data, test_data = divide_indices_into_test_and_train(data_indices, 2, num_batches)
        print(train_data)  # Output: [1, 2, 3, 4, 5, 6]
        print(test_data)   # Output: [7, 8, 9, 10]
    """
    batch_size = len(data_indices) // num_batches
    start_index = batch_number * batch_size
    end_index = start_index + batch_size

    if batch_number == num_batches - 1:
        test_data = data_indices[start_index:]
    else:
        test_data = data_indices[start_index:end_index]

    train_data = np.concatenate((data_indices[:start_index], data_indices[end_index:]))

    return train_data, test_data

def all_hospital_indexes_to_train_and_test_dicts(all_indexes_list):
    """
    Divide the indexes of data for each hospital into training and testing sets and store them in dictionaries.

    Parameters:
        all_indexes_list (list): A list of index lists for each hospital.

    Returns:
        tuple: A tuple containing two dictionaries. The first dictionary contains the training data indices for each hospital, and the second dictionary contains the testing data indices for each hospital.

    Example:
        all_indexes_list = [[1, 2, 3, 4, 5, 6, 7, 8, 9, 10], [11, 12, 13, 14, 15, 16, 17, 18, 19, 20]]
        train_dict, test_dict = all_hospital_indexes_to_train_and_test_dicts(all_indexes_list)
        print(train_dict)
        # Output: {0:  [3, 4, 5, 6, 7, 8, 9, 10], 1: [13, 14, 15, 16, 17, 18, 19, 20]}
        print(test_dict)
        # Output: {0: [1, 2], 1: [11, 12]}
    """
    data_dict = {}
    test_dict = {}

    for hospital_id, hospital_idxs in enumerate(all_indexes_list):
        # Split the data indices into training and testing sets using divide_indices_into_test_and_train function
        # The 1 and 5 values means that we want to split the data into 5 batches and take the first for the testing and keep the other four for training
        data_dict[hospital_id], test_dict[hospital_id] = divide_indices_into_test_and_train(hospital_idxs, 1, 5)

    return data_dict, test_dict

def get_train_and_test_idxs_for_server(datasets_len):
  indexes = []
  for ds_len in datasets_len:
    # The 1 and 5 values means that we want to split the data into 5 batches and take the first for the testing and keep the other four for training
    indexes.append(divide_indices_into_test_and_train(list(range(ds_len)), 1, 5))
  return [train_and_test_indices_for_dataset[0] for train_and_test_indices_for_dataset in indexes], [train_and_test_indices_for_dataset[1] for train_and_test_indices_for_dataset in indexes]

def download_datasets_and_partition_the_datasets(config):
    """
    Downloads the datasets specified in the config file.
    And it partitions them into train and test datasets for each hospital as well as for the server
    """
    logger = create_logger(config["log_path"])
    if "seed" in config.keys():
        seed = config["seed"]
        # Set the random seed for numpy
        np.random.seed(seed)
        # Set the random seed for Python's built-in random module
        random.seed(seed)

    # We define our data transformation
    resize = transforms.Resize((32, 32)) # Original size of the images is 28x28 pixels, we change it to 32x32 so it's compatible with our model

    # This class implements a grayscale transformation
    class ToSingleChannel(object):
        def __call__(self, img):
            return TF.rgb_to_grayscale(img)

    # We merge all of the transformations into one for ease of use
    # transforms.ToTensor() takes a numpy array and returns a pytorch.tensor, we use pytorch.tensors for our training
    data_transformation = transforms.Compose([ToSingleChannel(),resize, transforms.ToTensor()])
    
    # Downloading and paritioning the data
    dataset_names = config['datasets']
    paddings = {ds_name: len(INFO[ds_name]["label"]) for ds_name in dataset_names}
    datasets_classes = [getattr(medmnist, INFO[ds_name]["python_class"]) for ds_name in dataset_names]
    datasets = []
    server_datasets = []
    current_padding = 0
    hospital_indices = []
    size_per_dataset = config['size_per_dataset']

    # Get either the default or the specified data directory
    if 'data_directory' in config.keys():
        data_directory = config['data_directory']
    else:
        if sys.platform.startswith('linux'):
            data_directory = '/root/.medmnist'
        elif sys.platform.startswith('win'):
            data_directory = f'{os.getenv("SystemDrive")}\\Users\\{os.environ.get("USERNAME")}\\.medmnist'
        else:
            logger.critical('Unknow platform: %s' % sys.platform)
    if not os.path.exists(data_directory):
        logger.info("Creating data directory: %s" % data_directory)
        os.makedirs(data_directory)

    logger.info(f"Downloading the datasets {dataset_names} and partitioning them for the server and for the hospitals") 
    for ds_name, dataset in zip(dataset_names, datasets_classes):        
        datasets.append([*dataset(root=data_directory, download=True, split="train",transform=data_transformation, as_rgb=False),
                        *dataset(root=data_directory, download=True, split="test",transform=data_transformation, as_rgb=False),
                        *dataset(root=data_directory, download=True, split="val",transform=data_transformation, as_rgb=False)][:size_per_dataset])
        logger.info("downloaded dataset %s" % ds_name)
        # shuffle the dataset
        random.shuffle(datasets[-1])
        # add current padding to each label
        datasets[-1] = [(img, label+current_padding) for img, label in datasets[-1]]
        
        # update the current padding
        current_padding += paddings[ds_name]
        twenty_percent_index = int(len(datasets[-1]) * 0.2)
        server_datasets.append(datasets[-1][:twenty_percent_index])
        
        
        datasets[-1] = datasets[-1][twenty_percent_index:]
        
        # Divide this dataset into 19 non-iid values
        hospital_indices += list(non_iid_partition(datasets[-1], 10).values())
        logger.info(f"Partitioned {ds_name} for server and hospitals")
    return datasets, server_datasets, hospital_indices

def number_of_classes_from_datasets(datasets_names):
    return sum([len(INFO[ds_name]["label"]) for ds_name in datasets_names])
