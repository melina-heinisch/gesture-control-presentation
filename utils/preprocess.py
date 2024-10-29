import sys
import pandas as pd
import numpy as np
sys.path.append('..')
from utils.import_data import get_all_data
import operator

def preprocess_and_save_data(features, chunk_size = 15, subfolder_name =""):
    data = get_all_data(True)
    #features = ["timestamp",
                # "right_mouth_x", "right_mouth_y", "right_mouth_z",
                # "left_shoulder_x", "left_shoulder_y", "left_shoulder_z", 
                # "right_shoulder_x", "right_shoulder_y", "right_shoulder_z", 
                # "left_elbow_x", "left_elbow_y", "left_elbow_z", 
                # "right_elbow_x", "right_elbow_y", "right_elbow_z", 
                # "left_thumb_x", "left_thumb_y", "left_thumb_z",
                # "right_thumb_x", "right_thumb_y", "right_thumb_z",
                # "ground_truth"]
    data = data[features]
    data = data.set_index([pd.Index(range(0,0+len(data)))])

    #chunk_size = 15
    gestures_chunk, idle = get_gesture_chunks(data,chunk_size)
    separated_idle = get_idle_as_array(idle)
    idle_chunk = get_idle_chunk(separated_idle, chunk_size)

    flattened_gesture, Y_gesture = get_final_data(gestures_chunk)
    flattened_idle, Y_idle = get_final_data(idle_chunk)

    flattened_data = np.array(flattened_gesture + flattened_idle)
    Y = np.array(Y_gesture + Y_idle)

    # shuffle and split
    flattened_data, Y = unison_shuffled_copies(flattened_data, Y) 
    X_train, X_validate, X_test, Y_train, Y_validate, Y_test = split_data_evenly(flattened_data, Y)

    # shuffle again
    X_train, Y_train = unison_shuffled_copies(X_train, Y_train)
    X_validate, Y_validate = unison_shuffled_copies(X_validate, Y_validate)
    X_test, Y_test = unison_shuffled_copies(X_test, Y_test)

    if (subfolder_name != ""):
        subfolder_name = subfolder_name + "/"
    pd.DataFrame(X_train).to_csv(f"preprocessed_data/{subfolder_name}data_train.csv", index=False)
    pd.DataFrame(X_validate).to_csv(f"preprocessed_data/{subfolder_name}data_validate.csv", index=False)
    pd.DataFrame(X_test).to_csv(f"preprocessed_data/{subfolder_name}data_test.csv", index=False)
    pd.DataFrame(Y_train).to_csv(f"preprocessed_data/{subfolder_name}y_train.csv", index = False)
    pd.DataFrame(Y_validate).to_csv(f"preprocessed_data/{subfolder_name}y_validate.csv", index = False)
    pd.DataFrame(Y_test).to_csv(f"preprocessed_data/{subfolder_name}y_test.csv", index = False)

def preprocess_and_save_combined(relate, chunk_size = 15, subfolder_name =""):
    data_all = get_all_data(True)
    data = data_all[['timestamp', 'ground_truth']].copy()

    # 19 features
    data['left_thumb_shoulder_dif_x'] = relate(data_all['left_thumb_x'],data_all['left_shoulder_x'])
    data['left_thumb_shoulder_dif_y'] = relate(data_all['left_thumb_y'],data_all['left_shoulder_y']) 
    data['left_thumb_shoulder_dif_z'] = relate(data_all['left_thumb_z'],data_all['left_shoulder_z'])
    data['right_thumb_shoulder_dif_x'] = relate(data_all['right_thumb_x'],data_all['right_shoulder_x'])
    data['right_thumb_shoulder_dif_y'] = relate(data_all['right_thumb_y'],data_all['right_shoulder_y'])
    data['right_thumb_shoulder_dif_z'] = relate(data_all['right_thumb_z'],data_all['right_shoulder_z'])
    data['left_thumb_elbow_dif_x'] =  relate(data_all['left_thumb_x'],data_all['left_elbow_x'])
    data['left_thumb_elbow_dif_y'] = relate(data_all[ 'left_thumb_y' ],data_all[ 'left_elbow_y'])
    data['left_thumb_elbow_dif_z'] = relate(data_all['left_thumb_z' ],data_all['left_elbow_z'])
    data['right_thumb_elbow_dif_x'] = relate(data_all['right_thumb_x'],data_all['right_elbow_x'])
    data['right_thumb_elbow_dif_y'] = relate(data_all['right_thumb_y'],data_all['right_elbow_y']) 
    data['right_thumb_elbow_dif_z'] = relate(data_all['right_thumb_z'],data_all['right_elbow_z'])
    data['left_right_thumb_dif_x'] = relate(data_all[ 'left_thumb_x'],data_all[ 'right_thumb_x'])
    data['left_right_thumb_dif_y'] = relate(data_all['left_thumb_y' ],data_all['right_thumb_y'])
    data['left_right_thumb_dif_z'] = relate(data_all[ 'left_thumb_z'],data_all['right_thumb_z'])
    data['left_right_elbow_dif_x'] = relate(data_all['left_elbow_x'],data_all[ 'right_elbow_x'])
    data['left_right_elbow_dif_y'] = relate(data_all['left_elbow_y'],data_all['right_elbow_y'])
    data['left_right_elbow_dif_z'] = relate(data_all['left_elbow_z' ],data_all['right_elbow_z'])
    data['right_mouth_z'] = data_all['right_mouth_z']

    data = data.set_index([pd.Index(range(0,0+len(data_all)))])

    gestures_chunk, idle = get_gesture_chunks(data,chunk_size)
    separated_idle = get_idle_as_array(idle)
    idle_chunk = get_idle_chunk(separated_idle, chunk_size)

    flattened_gesture, Y_gesture = get_final_data(gestures_chunk)
    flattened_idle, Y_idle = get_final_data(idle_chunk)

    flattened_data = np.array(flattened_gesture + flattened_idle)
    Y = np.array(Y_gesture + Y_idle)

    # shuffle and split
    flattened_data, Y = unison_shuffled_copies(flattened_data, Y) 
    X_train, X_validate, X_test, Y_train, Y_validate, Y_test = split_data_evenly(flattened_data, Y)

    # shuffle again
    X_train, Y_train = unison_shuffled_copies(X_train, Y_train)
    X_validate, Y_validate = unison_shuffled_copies(X_validate, Y_validate)
    X_test, Y_test = unison_shuffled_copies(X_test, Y_test)

    if (subfolder_name != ""):
        subfolder_name = subfolder_name + "/"
    pd.DataFrame(X_train).to_csv(f"preprocessed_data/{subfolder_name}data_train.csv", index=False)
    pd.DataFrame(X_validate).to_csv(f"preprocessed_data/{subfolder_name}data_validate.csv", index=False)
    pd.DataFrame(X_test).to_csv(f"preprocessed_data/{subfolder_name}data_test.csv", index=False)
    pd.DataFrame(Y_train).to_csv(f"preprocessed_data/{subfolder_name}y_train.csv", index = False)
    pd.DataFrame(Y_validate).to_csv(f"preprocessed_data/{subfolder_name}y_validate.csv", index = False)
    pd.DataFrame(Y_test).to_csv(f"preprocessed_data/{subfolder_name}y_test.csv", index = False)


def get_gesture_chunks(input, chunk_size):
    separate_gestures = []
    previous_movement = "idle"
    j = -1
    idle = input.copy()
    for i in range(len(input)):
        if i < j:
            previous_movement = input.iloc[i]['ground_truth']
            continue
        if (previous_movement == "idle" and input.iloc[i]['ground_truth'] != 'idle'):
            j = i+chunk_size
            chunk = input.iloc[i:j]
            separate_gestures.append(chunk)
            idle.drop(chunk.index, axis=0, inplace=True)
        
        previous_movement = input.iloc[i]['ground_truth']
    
    return separate_gestures, idle

def get_idle_as_array(input):
    separate_idle = []
    prev_time = input.iloc[0]["timestamp"]
    current_idle = []
    for elem in input.iterrows():
        diff = abs(prev_time - elem[1]["timestamp"])
        diff_ms = diff.total_seconds() * 1000
        if(diff_ms < 50):
            current_idle.append(elem[1].to_frame().T)
        else:
            if current_idle:
                separate_idle.append(pd.concat(current_idle))
                current_idle = []
        prev_time = elem[1]["timestamp"]

    return separate_idle

def get_idle_chunk(input, chunk_size):
    final_idle = []
    for i in range(len(input)):
        if(len(input[i]) >= chunk_size):
            chunk = input[i].iloc[0:chunk_size]
            final_idle.append(chunk)
    return final_idle

def split_data_evenly(X, Y):
    X_train = []
    X_validate = []
    X_test = []
    Y_train = []
    Y_validate = []
    Y_test = []

    for gesture in np.unique(Y):
        gesture_index = np.where(Y == gesture)
        data_for_gesture = X[gesture_index]
        Y_for_gesture = Y[gesture_index]

        train_validation_split = round(Y_for_gesture.shape[0]*0.6)
        validation_test_split = round(Y_for_gesture.shape[0]*0.8)

        X_train.append(data_for_gesture[:train_validation_split:])
        X_validate.append(data_for_gesture[train_validation_split:validation_test_split:])
        X_test.append(data_for_gesture[validation_test_split::])
        Y_train.append(Y_for_gesture[:train_validation_split:])
        Y_validate.append(Y_for_gesture[train_validation_split:validation_test_split:])
        Y_test.append(Y_for_gesture[validation_test_split::])

    #comment this block when generating for test mode
    X_train = np.array(X_train)
    X_validate = np.array(X_validate)
    X_test = np.array(X_test)
    Y_train = np.array(Y_train)
    Y_validate = np.array(Y_validate)
    Y_test = np.array(Y_test)
    
    # flatten
    X_train = flatten_array_during_data_split(X_train)
    X_validate = flatten_array_during_data_split(X_validate)
    X_test = flatten_array_during_data_split(X_test)
    Y_train = np.hstack(Y_train)
    Y_validate = np.hstack(Y_validate)
    Y_test = np.hstack(Y_test)

    return X_train, X_validate, X_test, Y_train, Y_validate, Y_test

def flatten_array_during_data_split(input):
    return np.concatenate((input[0], input[1] , input[2], input[3], input[4], input[5], input[6], input[7], input[8], input[9], input[10]), axis=0)
    #return np.concatenate((input[0], input[1] , input[2], input[3]))

def get_final_data(input):
    X = []
    Y = []

    for df in input:
        Y.append(df['ground_truth'].values[0])
        df_without_y = df.loc[:, (df.columns != 'ground_truth') & (df.columns != 'timestamp')]
        result = df_without_y.to_numpy().flatten()
        X.append(result)

    return X, Y

# via https://stackoverflow.com/questions/4601373/better-way-to-shuffle-two-numpy-arrays-in-unison
def unison_shuffled_copies(a, b):
    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    return a[p], b[p]