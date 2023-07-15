'''
## Configuration file

* Algorithm
* Datasets
* Datasets size
* Number of nodes (hospitals)
* Batch size
* Learning rate
* Local epochs
* Global epochs (If needed)
* Data directory
* Results directory
* Verbose logs or not
* Participation percentage
* Mu (If needed)
* IID /  non-IID partition
'''
config = {
    "algorithm": "fedavg",
    "results_path": "./results/",
    "datasets": ["octmnist", "bloodmnist"], # , "organamnist", "pathmnist", "tissuemnist
    "model": "vgg11",
    "size_per_dataset": 22000,
    "data_directory": "./data/",
    # "number_of_hospitals": 0,
    "batch_size": 50,
    "learning_rate": 0.00001,
    "local_epochs": 2,
    "global_epochs": 4,
    "participation_percent": 0.1
}