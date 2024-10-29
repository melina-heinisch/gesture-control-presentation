import numpy as np
import pandas as pd
import pathlib

from machine_learning_framework import NeuralNet, tanh, gradient_descent, StandardScaler

here = pathlib.Path(__file__).parent

column_names = ['timestamp',
                'left_thumb_shoulder_dif_x','left_thumb_shoulder_dif_y','left_thumb_shoulder_dif_z',
                'right_thumb_shoulder_dif_x','right_thumb_shoulder_dif_y','right_thumb_shoulder_dif_z',
                'left_thumb_elbow_dif_x','left_thumb_elbow_dif_y','left_thumb_elbow_dif_z',
                'right_thumb_elbow_dif_x','right_thumb_elbow_dif_y','right_thumb_elbow_dif_z',
                'left_right_thumb_dif_x','left_right_thumb_dif_y','left_right_thumb_dif_z',
                'left_right_elbow_dif_x','left_right_elbow_dif_y','left_right_elbow_dif_z',
                'right_mouth_z']

pause_after_emit = False


def flatten(data,chunk_size):
    data = data[0:chunk_size]
    return data.to_numpy().flatten()

def interpolate(data):
    data["timestamp"] = pd.to_timedelta(data["timestamp"], unit="ms")
    data["timestamp"]= data["timestamp"].dt.round(freq="L")
    data = data.set_index("timestamp")
    data = data.resample('1ms').interpolate()
    data = data.resample('33ms').mean()
    return data

def interpret_prediction(prediction):
    index = np.argmax(prediction, axis=1)
    max = np.max(prediction)
    labels = ['idle', 'rotate', 'swipe_left', 'swipe_right']
    return labels[index[0]], max

def define_variables(here):
    global column_names
    w_container = np.load(here.joinpath('trained_weights/weights.npz'))
    weights = [w_container["w0"], w_container["w1"], w_container["w2"]]
    b_container = np.load(here.joinpath('trained_weights/biases.npz'))
    biases = [b_container["b0"], b_container["b1"], b_container["b2"]]
    layers_size = [285, 76, 38, 4]
    neural_net = NeuralNet(layers_size, NeuralNet.Type.MUTILCLASS_CLASSIFICATION, tanh, gradient_descent, weights, biases)

    X = pd.read_csv(here.joinpath('data_train.csv'))
    X = X.to_numpy()
    scaler = StandardScaler()
    scaler.fit(X)

    return neural_net, scaler

def get_data(frames):
    global column_names

    data = pd.DataFrame(columns = column_names)
    data.loc[0] = frames[column_names].copy()
    data["timestamp"] = pd.to_timedelta(data["timestamp"], unit="ms")
    data = data.set_index("timestamp")

    return data

def get_calc_data(frames):
    global column_names

    data = pd.DataFrame(columns = column_names)

    data['timestamp'] = frames['timestamp']
    data['right_mouth_z'] = frames['right_mouth_z']
    data['left_thumb_shoulder_dif_x'] = frames['left_thumb_x']-frames['left_shoulder_x']
    data['left_thumb_shoulder_dif_y'] = frames['left_thumb_y']-frames['left_shoulder_y'] 
    data['left_thumb_shoulder_dif_z'] = frames['left_thumb_z']-frames['left_shoulder_z']
    data['right_thumb_shoulder_dif_x'] = frames['right_thumb_x']-frames['right_shoulder_x']
    data['right_thumb_shoulder_dif_y'] = frames['right_thumb_y']-frames['right_shoulder_y']
    data['right_thumb_shoulder_dif_z'] = frames['right_thumb_z']-frames['right_shoulder_z']
    data['left_thumb_elbow_dif_x'] =  frames['left_thumb_x']-frames['left_elbow_x']
    data['left_thumb_elbow_dif_y'] = frames[ 'left_thumb_y' ]-frames[ 'left_elbow_y']
    data['left_thumb_elbow_dif_z'] = frames['left_thumb_z' ]-frames['left_elbow_z']
    data['right_thumb_elbow_dif_x'] = frames['right_thumb_x']-frames['right_elbow_x']
    data['right_thumb_elbow_dif_y'] = frames['right_thumb_y']-frames['right_elbow_y'] 
    data['right_thumb_elbow_dif_z'] = frames['right_thumb_z']-frames['right_elbow_z']
    data['left_right_thumb_dif_x'] = frames[ 'left_thumb_x']-frames[ 'right_thumb_x']
    data['left_right_thumb_dif_y'] = frames['left_thumb_y' ]-frames['right_thumb_y']
    data['left_right_thumb_dif_z'] = frames[ 'left_thumb_z']-frames['right_thumb_z']
    data['left_right_elbow_dif_x'] = frames['left_elbow_x']-frames[ 'right_elbow_x']
    data['left_right_elbow_dif_y'] = frames['left_elbow_y']-frames['right_elbow_y']
    data['left_right_elbow_dif_z'] = frames['left_elbow_z' ]-frames['right_elbow_z']

    return data

def get_decision(predictions):
    global pause_after_emit
    np_pred = np.array(predictions)

    unique, counts = np.unique(np_pred, return_counts=True)
    elements = dict(zip(unique, counts))
    if elements.get("idle",0) >= 20 and pause_after_emit:
        pause_after_emit = False
        return "idle", True
    if not pause_after_emit:
        for key, value in elements.items():
            if key != "idle":
                if value >= 4:
                    pause_after_emit = True
                    return key, True
        return "idle", False
    else:
        return "idle", False
