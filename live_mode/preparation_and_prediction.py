import numpy as np
import pandas as pd
import asyncio
import pathlib

from machine_learning_framework import NeuralNet, tanh, gradient_descent, StandardScaler

import os
here = pathlib.Path(__file__).parent
os.chdir(here.parent.joinpath('./utils/'))

import sys
sys.path.append('..')
from utils.gesture import Gesture

features = ["right_mouth", "left_shoulder", "right_shoulder", "left_elbow", "right_elbow", "left_thumb", "right_thumb"]

column_names_raw = ["timestamp",
                "right_mouth_z",
                "left_shoulder_x", "left_shoulder_y", "left_shoulder_z",
                "right_shoulder_x","right_shoulder_y","right_shoulder_z",
                "left_elbow_x", "left_elbow_y", "left_elbow_z", 
                "right_elbow_x", "right_elbow_y", "right_elbow_z", 
                "left_thumb_x", "left_thumb_y", "left_thumb_z",
                "right_thumb_x", "right_thumb_y", "right_thumb_z"]

column_names = ['timestamp',
                'left_thumb_shoulder_dif_x','left_thumb_shoulder_dif_y','left_thumb_shoulder_dif_z',
                'right_thumb_shoulder_dif_x','right_thumb_shoulder_dif_y','right_thumb_shoulder_dif_z',
                'left_thumb_elbow_dif_x','left_thumb_elbow_dif_y','left_thumb_elbow_dif_z',
                'right_thumb_elbow_dif_x','right_thumb_elbow_dif_y','right_thumb_elbow_dif_z',
                'left_right_thumb_dif_x','left_right_thumb_dif_y','left_right_thumb_dif_z',
                'left_right_elbow_dif_x','left_right_elbow_dif_y','left_right_elbow_dif_z',
                'right_mouth_z']

pause_after_emit = False

def interpolate(data):
    data.index = data.index.round(freq="L")
    data = data.resample('1ms').interpolate()
    data = data.resample('33ms').mean()
    return data

def flatten(data,chunk_size):
    data = data[0:chunk_size]
    return data.to_numpy().flatten()

def interpret_prediction(prediction):
    index = np.argmax(prediction, axis=1)
    max = np.max(prediction)
    labels = ['down', 'flip_table', 'idle', 'pinch', 'rotate_left','rotate_right', 'spin', 'spread', 'swipe_left', 'swipe_right','up']
    return labels[index[0]], max

def define_variables(here):
    global column_names
    w_container = np.load(here.joinpath('trained_weights/weights.npz'))
    weights = [w_container["w0"], w_container["w1"], w_container["w2"]]
    b_container = np.load(here.joinpath('trained_weights/biases.npz'))
    biases = [b_container["b0"], b_container["b1"], b_container["b2"]]
    layer_size = [570, 76, 38, 11]
    neural_net = NeuralNet(layer_size, NeuralNet.Type.MUTILCLASS_CLASSIFICATION, tanh, gradient_descent, weights, biases)

    X = pd.read_csv(here.joinpath('data_train.csv'))
    X = X.to_numpy()
    scaler = StandardScaler()
    scaler.fit(X)

    data = pd.DataFrame(columns=column_names)
    data = data.set_index("timestamp")

    return neural_net, scaler, data

def calc_data_row(results, cap, cv2, KEYPOINT_NAMES):
    global features
    global column_names
    global column_names_raw
    raw = [cap.get(cv2.CAP_PROP_POS_MSEC)]

    for joint_name in features: # you can choose any joint listed in `KEYPOINT_NAMES`
        if(joint_name == "right_mouth"):
            raw += [results.pose_landmarks.landmark[KEYPOINT_NAMES.index(joint_name)].z]
        else:
            raw += [results.pose_landmarks.landmark[KEYPOINT_NAMES.index(joint_name)].x,
                    results.pose_landmarks.landmark[KEYPOINT_NAMES.index(joint_name)].y,
                    results.pose_landmarks.landmark[KEYPOINT_NAMES.index(joint_name)].z]

    raw_df = pd.DataFrame(columns = column_names_raw)
    raw_df.loc[0] = raw
    final_df = pd.DataFrame(columns = column_names)

    final_df['timestamp'] = raw_df['timestamp']
    final_df['right_mouth_z'] = raw_df['right_mouth_z']
    final_df['left_thumb_shoulder_dif_x'] = raw_df['left_thumb_x']-raw_df['left_shoulder_x']
    final_df['left_thumb_shoulder_dif_y'] = raw_df['left_thumb_y']-raw_df['left_shoulder_y']
    final_df['left_thumb_shoulder_dif_z'] = raw_df['left_thumb_z']-raw_df['left_shoulder_z']
    final_df['right_thumb_shoulder_dif_x'] = raw_df['right_thumb_x']-raw_df['right_shoulder_x']
    final_df['right_thumb_shoulder_dif_y'] = raw_df['right_thumb_y']-raw_df['right_shoulder_y']
    final_df['right_thumb_shoulder_dif_z'] = raw_df['right_thumb_z']-raw_df['right_shoulder_z']
    final_df['left_thumb_elbow_dif_x'] =  raw_df['left_thumb_x']-raw_df['left_elbow_x']
    final_df['left_thumb_elbow_dif_y'] = raw_df[ 'left_thumb_y' ]-raw_df[ 'left_elbow_y']
    final_df['left_thumb_elbow_dif_z'] = raw_df['left_thumb_z' ]-raw_df['left_elbow_z']
    final_df['right_thumb_elbow_dif_x'] = raw_df['right_thumb_x']-raw_df['right_elbow_x']
    final_df['right_thumb_elbow_dif_y'] = raw_df['right_thumb_y']-raw_df['right_elbow_y']
    final_df['right_thumb_elbow_dif_z'] = raw_df['right_thumb_z']-raw_df['right_elbow_z']
    final_df['left_right_thumb_dif_x'] = raw_df[ 'left_thumb_x']-raw_df[ 'right_thumb_x']
    final_df['left_right_thumb_dif_y'] = raw_df['left_thumb_y' ]-raw_df['right_thumb_y']
    final_df['left_right_thumb_dif_z'] = raw_df[ 'left_thumb_z']-raw_df['right_thumb_z']
    final_df['left_right_elbow_dif_x'] = raw_df['left_elbow_x']-raw_df[ 'right_elbow_x']
    final_df['left_right_elbow_dif_y'] = raw_df['left_elbow_y']-raw_df['right_elbow_y']
    final_df['left_right_elbow_dif_z'] = raw_df['left_elbow_z' ]-raw_df['right_elbow_z']

    final_df["timestamp"] = pd.to_timedelta(final_df["timestamp"], unit="ms")
    final_df = final_df.set_index("timestamp")

    return final_df

async def decision_emit(ws,predictions):
    global pause_after_emit
    np_pred = np.array(predictions)

    unique, counts = np.unique(np_pred, return_counts=True)
    elements = dict(zip(unique, counts))
    if elements.get("idle",0) >= 15 and pause_after_emit:
        pause_after_emit = False
        return True
    if not pause_after_emit:
        for key, value in elements.items():
            if key != "idle":
                if value >= 5:
                    await emit_event(ws,key)
                    pause_after_emit = True
                    return True

async def emit_event(ws, gesture):
    if gesture == Gesture.SWIPE_RIGHT.value:
        print("emitting 'right'")
        await ws.send("right")
        await asyncio.sleep(0.001)
    elif gesture == Gesture.SWIPE_LEFT.value:
        print("emitting 'right'")
        await ws.send("left")
        await asyncio.sleep(0.001)
    elif gesture == Gesture.ROTATE_RIGHT.value:
        print("emitting 'rotate_right'")
        await ws.send("rotate_right")
        await asyncio.sleep(0.001)
    elif gesture == Gesture.ROTATE_LEFT.value:
        print("emitting 'rotate_left'")
        await ws.send("rotate_left")
        await asyncio.sleep(0.001)
    elif gesture == Gesture.PINCH.value:
        print("emitting 'zoom out'")
        await ws.send("zoom_out")
        await asyncio.sleep(0.001)
    elif gesture == Gesture.SPREAD.value:
        print("emitting 'zoom_in'")
        await ws.send("zoom_in")
        await asyncio.sleep(0.001)
    elif gesture == Gesture.UP.value:
        print("emitting 'up'")
        await ws.send("up")
        await asyncio.sleep(0.001)
    elif gesture == Gesture.DOWN.value:
        print("emitting 'down'")
        await ws.send("down")
        await asyncio.sleep(0.001)
    elif gesture == Gesture.FLIP_TABLE.value:
        print("emitting 'overview'")
        await ws.send("overview")
        await asyncio.sleep(0.001)
    elif gesture == Gesture.SPIN.value:
        print("emitting 'jump_end'")
        await ws.send("jump_end")
        await asyncio.sleep(0.001)