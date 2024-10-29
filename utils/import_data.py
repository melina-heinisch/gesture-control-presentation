import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

body_part_dict = {
    "nose": ["nose_x", "nose_y", "nose_z"],    
    "left_eye_inner": ["left_eye_inner_x", "left_eye_inner_y", "left_eye_inner_z"],
    "left_eye": ["left_eye_x", "left_eye_y", "left_eye_z"],
    "left_eye_outer": ["left_eye_outer_x", "left_eye_outer_y", "left_eye_outer_z"],
    "right_eye_inner": ["right_eye_inner_x", "right_eye_inner_y", "right_eye_inner_z"],
    "right_eye": ["right_eye_x", "right_eye_y", "right_eye_z"],
    "right_eye_outer": ["right_eye_outer_x", "right_eye_outer_y", "right_eye_outer_z"],
    "left_ear": ["left_ear_x", "left_ear_y", "left_ear_z"],
    "right_ear": ["right_ear_x", "right_ear_y", "right_ear_z"],
    "left_mouth": ["left_mouth_x", "left_mouth_y", "left_mouth_z"],
    "right_mouth": ["right_mouth_x", "right_mouth_y", "right_mouth_z"],
    "left_shoulder": ["left_shoulder_x", "left_shoulder_y", "left_shoulder_z"],
    "right_shoulder": ["right_shoulder_x", "right_shoulder_y", "right_shoulder_z"],
    "left_elbow": ["left_elbow_x", "left_elbow_y", "left_elbow_z"],
    "right_elbow": ["right_elbow_x", "right_elbow_y", "right_elbow_z"],
    "left_wrist": ["left_wrist_x", "left_wrist_y", "left_wrist_z"],
    "right_wrist": ["right_wrist_x", "right_wrist_y", "right_wrist_z"],
    "left_pinky": ["left_pinky_x", "left_pinky_y", "left_pinky_z"],
    "right_pinky": ["right_pinky_x", "right_pinky_y", "right_pinky_z"],
    "left_index": ["left_index_x", "left_index_y", "left_index_z"],
    "right_index": ["right_index_x", "right_index_y", "right_index_z"],
    "left_thumb": ["left_thumb_x", "left_thumb_y", "left_thumb_z"],
    "right_thumb": ["right_thumb_x", "right_thumb_y", "right_thumb_z"],
    "left_hip": ["left_hip_x", "left_hip_y", "left_hip_z"],
    "right_hip": ["right_hip_x", "right_hip_y", "right_hip_z"],
    "left_knee": ["left_knee_x", "left_knee_y", "left_knee_z"],
    "right_knee": ["right_knee_x", "right_knee_y", "right_knee_z"],
    "left_ankle": ["left_ankle_x", "left_ankle_y", "left_ankle_z"],
    "right_ankle": ["right_ankle_x", "right_ankle_y", "right_ankle_z"],
    "left_heel": ["left_heel_x", "left_heel_y", "left_heel_z"],
    "right_heel": ["right_heel_x", "right_heel_y", "right_heel_z"],
    "left_foot_index": ["left_foot_index_x", "left_foot_index_y", "left_foot_index_z"],
    "right_foot_index": ["right_foot_index_x", "right_foot_index_y", "right_foot_index_z"]
}

# returns an array with all data
def get_all_data(useIdMethod = False):
    if useIdMethod:
        method = csv_file_to_dataframe_id
    else:
        method = csv_file_to_dataframe

    files = pd.read_csv("filenames.csv")
    files_array = files.to_numpy()
    dataframe = list()

    for file in files_array:
        file = str(file)
        file = file[2:]
        file = file[:-2]
        dataframe.append(method(file))

    return pd.concat(dataframe)

# returns an array with all data of the selected gesture
def get_all_data_by_gesture(gesture):
    files = pd.read_csv("filenames.csv")
    files_array = files.to_numpy()
    dataframe = list()

    for file in files_array:
        file = str(file)
        if(file.find(gesture) != -1):
            file = file[2:]
            file = file[:-2]
            dataframe.append(csv_file_to_dataframe(file) )

    return np.array(dataframe, dtype=object)


def csv_file_to_dataframe(frames_file_path):
  frames = pd.read_csv(frames_file_path)

  # set timestamp column as index, so we can do timerange based selections
  frames["timestamp"] = pd.to_timedelta(frames["timestamp"], unit="ms")
  frames = frames.set_index("timestamp")
  frames.index = frames.index.rename("timestamp")
  return frames

def csv_file_to_dataframe_id(frames_file_path):
    frames = pd.read_csv(frames_file_path)

    # set timestamp column as index, so we can do timerange based selections
    frames["timestamp"] = pd.to_timedelta(frames["timestamp"], unit="ms")
    return frames

# get dataframe to do subplots
def get_all_data_by_gesture_and_body_part(gesture, body_part): 
    all_frames_concat = get_all_data()
    return all_frames_concat[body_part_dict[body_part]][all_frames_concat["ground_truth"] == gesture]

def get_gesture_length(gesture):
    video_data = get_all_data_by_gesture(gesture)
    
    length_of_gestures = []
    for df in video_data:
        start_times, stop_times = get_start_and_stop_time(df, gesture)

        # iteration über alle einzelnen Gesten
        for i in range(len(start_times)):
            start = start_times[i] 
            stop = stop_times[i]
            #heraussuchen der frames, in denen die Geste ausgeführt wird
            frames_of_one_gesture = df[(df.index >= start) & (df.index <= stop)]
            length_of_gestures.append(len(frames_of_one_gesture))

    return length_of_gestures

def get_start_and_stop_time(video_frames, movement):
    start = []
    stop = []

    previous_movement = "idle"
    previous_timestamp = ""
    for frame in video_frames.iterrows():
        if (previous_movement == "idle" and frame[1]['ground_truth'] == movement):
            start.append(frame[0])
        elif (previous_movement == movement and frame[1]['ground_truth'] == "idle"):
            # append timstamp from frame before to stop
            stop.append(previous_timestamp)
        
        previous_movement = frame[1]['ground_truth']
        previous_timestamp = frame[0]

    return start, stop

def get_mean_gesture_plot(gesture, body_part, start_time_modificator):
    video_data = get_all_data_by_gesture(gesture)
    
    for df in video_data:
        start_times, stop_times = get_start_and_stop_time(df, gesture)

        all_gesture_frames = []
        # iteration über alle einzelnen Gesten
        for i in range(len(start_times)):
            start = start_times[i] + start_time_modificator
            stop = stop_times[i]
            #heraussuchen der frames, in denen die Geste ausgeführt wird
            frames_of_one_gesture = df[(df.index >= start) & (df.index <= stop)][body_part_dict[body_part]]
            all_gesture_frames.append(frames_of_one_gesture)

        mean_gesture = []
        # über alle Frames der Bewegung interieren:
        # die längste Bewegung wird für die Länger der for-Schleife herausgesucht, 
        # aus jeder Bewegung wird das ensprechende Frame (1. / 2. / 3. etc) verwendet, um den jeweiligen Mittelwert zu bilden
        for frame_nr in range(len(max(all_gesture_frames, key=len))): # für jedes Frame
            current_frames = []
            for movement_nr in range(len(all_gesture_frames)): 
                try:
                    current_frames.append(all_gesture_frames[movement_nr].iloc[[frame_nr]])
                except IndexError:
                    pass
            mean_gesture.append(np.mean(current_frames, axis=0)[0]) # für jede Bewegung MW finden

    plt.plot(np.array(mean_gesture), label=[body_part+'_x', body_part+'_y', body_part+'_z'])
    plt.legend()
    plt.xlabel("Frames")
    plt.ylabel("Coordinates")
    plt.title(f"Mean of all {gesture} for {body_part}")


# returns an array with all data
def get_all_test_mode_data(useIdMethod = False):
    if useIdMethod:
        method = csv_file_to_dataframe_id
    else:
        method = csv_file_to_dataframe

    files = pd.read_csv("filenames_test_mode.csv")
    files_array = files.to_numpy()
    dataframe = list()

    for file in files_array:
        file = str(file)
        file = file[2:]
        file = file[:-2]
        dataframe.append(method(file))

    return pd.concat(dataframe)