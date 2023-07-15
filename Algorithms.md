# Algorithms

* FedAvg
* FedProx
* FedBN
* Sequential FL
* Clustered Sequential FL

# Components

## Main.py

* Reads the config file
* Selects appropriate server and hospital class
* Partitions data
* Sets up a logger and sets up a logging directory/file
* Creates plots and puts them in specified directory

## Servers

* Federated Averaging (normal) with and without server training
* Sequential FL
* Clustered Sequential FL
* FedBN with and without server training

Server calls the clients train method and later call their training function if server model training is possible

Testing on the global test data, needs to have other metrics such as sensitivity and specificity etc.. **FIX THE METRICS FUNCTION** 

## Hospitals

* Normal Hospital
  * Can computes a score if needed
  * Train function
  * Testing function gets called just after the training, metrics:
    * Accuracy
* FedProx hospital 
  * Train function with proximal term
  * Testing function gets called just after the training, metrics:
    * Accuracy

## Configuration file

* Algorithm
* Datasets
* Datasets size
* (omitted) Number of nodes (hospitals)
* Batch size
* Learning rate
* Local epochs
* Global epochs (If needed)
* Data directory
* Results directory
* Verbose logs or not
* Participation percentage
* Mu (If needed)
* (omitted) IID /  non-IID partition