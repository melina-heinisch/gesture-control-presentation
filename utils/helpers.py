import numpy as np

def interpret_predictions(prediction):
    indices = np.argmax(prediction, axis=1)
    return indices
