from sanic import Sanic
from sanic.response import html
import pathlib

import cv2
import mediapipe as mp
import yaml
import pandas as pd

from preparation_and_prediction import interpolate, flatten, interpret_prediction, decision_emit, define_variables, calc_data_row

here = pathlib.Path(__file__).parent
# ---------------------------- Slideshow Server prep ----------------------------------

slideshow_root_path = here.joinpath("slideshow")
app = Sanic("slideshow_server")
app.static("/static", slideshow_root_path)

@app.route("/")
async def index(request):
    return html(open(slideshow_root_path.joinpath("slideshow_ML.html"), "r").read())

@app.websocket("/events")
async def emitter(_request, ws):
    print("websocket connection opened")
    
    # ---------------------------- Live Video Feed Prep --------------------------------
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles
    mp_pose = mp.solutions.pose

    show_video = True
    show_data = True
    flip_image = False # when your webcam flips your image, you may need to re-flip it by setting this to True
    success = True

    cap = cv2.VideoCapture(index=0) 
    #cap = cv2.VideoCapture(filename=str(here.joinpath("../demo_data/team_example.mp4")))    # Video

    # the names of each joint ("keypoint") are defined in this yaml file:
    with open(here.joinpath("keypoint_mapping.yml"), "r") as yaml_file:
        mappings = yaml.safe_load(yaml_file)
        KEYPOINT_NAMES = mappings["face"]
        KEYPOINT_NAMES += mappings["body"]

    neural_net, scaler, data = define_variables(here)
    chunk_size = 30
    chunk = pd.DataFrame()
    predictions = []

    # ======================== Loop for collecing frames from live video ======================

    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        while cap.isOpened() and success:
            success, image = cap.read()
            if not success:
                break

            if flip_image:
                image = cv2.flip(image, 1)

            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
            results = pose.process(image)

            # Draw the pose annotation on the image
            if show_video:
                image.flags.writeable = True
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

                mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                        landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())
                cv2.imshow('MediaPipe Pose', image)

            # press ESC to stop the loop
            if cv2.waitKey(5) & 0xFF == 27:
                break

            # ===== read and process data =====
            if show_data and results.pose_landmarks is not None:
                # Get timestamp and joint coordinated and put them in a dataframe
                #current_df = get_data_row(results, cap, cv2, KEYPOINT_NAMES)
                current_df = calc_data_row(results, cap, cv2, KEYPOINT_NAMES)
                
                # Interpolate every two frames and add result to our data
                if len(data) >= 1:
                    last = data.tail(1)
                    result = interpolate(pd.concat([last, current_df]))
                    data = pd.concat([data, result])
                else:
                    data = pd.concat([data, current_df])

                # Once we have a big enough chunk, preprocess, predict and emit fitting event
                if(len(data) >= chunk_size):
                    chunk = flatten(data, chunk_size)
                    chunk = scaler.transform(chunk)
                    result = neural_net.predict(chunk)
                    prediction, confidence = interpret_prediction(result)
                    print(f"{prediction} with {confidence}")
                    if(confidence > 0.7 or prediction == "idle"):
                        predictions.append(prediction) 
                    else:
                        predictions.append("idle")
                    if await decision_emit(ws, predictions):
                        predictions = []
                    data = data.tail(29) # keeping all but the first frames
                
    cap.release()

if __name__ == "__main__":
    app.run(host="127.0.0.1", debug=True)
