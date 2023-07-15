import numpy as np

def compute_accuracy(y_pred, y_test) -> np.float32:
    """
    Computes the accuracy of the model.
    """
    return np.sum(np.array(y_pred).flatten() == np.array(y_test).flatten())

def calculate_hospital_score(loss_values):
    # returns the loss values if  the loss list contains only one element
    if len(loss_values) == 1:
       return loss_values[0]
    # Compute the rate of convergence
    x = np.arange(0, len(loss_values))
    y = np.array(loss_values)
    slope , _ = np.polyfit(x, y, deg=1)
    # You can customize this formula based on your preferences
    # Define a scoring formula that combines rate of convergence and stability
    return slope

def flatten(array):
  return [item for sublist in array for item in sublist]

def get_mean_accuracy_for_group(server, group, round):
  return np.mean([server.hospitals[i].data_frame["Accuracy"][round+1] for i in group])