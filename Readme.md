# Machine Learning Video Gesture Recognition for Presentation Control
This project contains the model to perform the live and test mode for gesture recognition. It uses the Machine Learning Framework developed by Melina Heinisch which can be seen [here](https://github.com/melina-heinisch/machine-learning-framework).

Demonstration Video can be seen [here](https://youtu.be/TdlGVBEaN40?si=UFzzpnB8GzJFU4dC).

# Gestures
The following gestures are implemented and can be used to control a `reveal.js` slideshow.

1. Swipe right: Swipe gesture with left arm to go to the next slide
2. Swipe left: Swipe gesture with right arm to go to the previous slide
3. Rotate right: Rotate gesture with right arm to rotate pictures clockwise.
4. Rotate left: Rotate gesture with left arm to rotate pictures counter-clockwise.
5. Spread: Perform a spread gesture to zoom in: both hands in front of your chest moving the right arm on the right side next to your head and the left arm on the left side next to your legs 
6. Pinch: Perform a pinch gesture to zoom out: left arm on the left side next to your head and the right arm on the right side next to your legs moving to the front of your chest
7. Up: Up gesture with right arm to navigate to the subslides.
8. Down: Down gesture with left arm to navigate between the subslides.
9. Flip Table: Move both hands to your head (as if you were flipping a table) to go to the overview of all slides / to go back to the selected slide.
10. Spin: Rotate both hands around each other in front of your chest to jump to the end of the slideshow.

# Setup
 * you need at least Python 3.7
1. Open your terminal
2. Switch to this directory
3. Install dependencies listed in `requirements.txt`, for example with `pip install -r requirements.txt`
4. Install the Machine Learning Framework with `pip install git+https://gitlab2.informatik.uni-wuerzburg.de/hci/teaching/courses/machine-learning/student-submissions/ws22/Team-1/machine-learning-framework.git`

# Live mode:
1. run the server with `python application.py`
2. open the chrome browser and visit http://localhost:8000 (it is important to use chrome, since the zoom functionality is not supported by other browsers)

# Performance Score / Test mode: 
The script `log_emitted_events_to_csv.py` reads a CSV-file and produces a new CSV-file containing a column `events` that contains the events your application registers for each frame.  
You can use the script like this:

    python performance_score/log_emitted_events_to_csv.py --input_csv=demo_data/demo_video_frames_rotate.csv

`calculator.py` determines the performance score.  
Call the script like this:

    python performance_score/calculator.py --events_csv=demo_data/emitted_events.csv --ground_truth_csv=demo_data/demo_video_csv_with_ground_truth_rotate.csv


# Contents

## Data
___
In `data` you can find the csv files we used for training our models.

## Data Exploration
___
There is one general `jupyter notebook` called `data_exploration`, in which we explored the data:
- dataframe of all data 
- amount of frames for each gesture
- confidence of the body parts (based on these values, we decided to explore the data only above the hip for each gesture)
- correlations between the body parts

We created a `jupyter notebook` for each gesture to explore the data:
- shortest, longest, mean and median gesture length
- plots for all samples separated by body part (x,y,z values)
- plots for mean gestures separated by body part (x,y,z values)

## Live mode
___
To start the live mode, you need to run the `application.py` script.
The structure of the slideshow can be found in `slideshow/slideshow_ML.html`. All helpers and event listeners are in the same directory.

The helpers for the live mode are in `preparation_and_prediciton.py`:
- `interpolate`: method to interpolate the data to 30 fps
- `flatten`: method to reduce the data to the chunk size and flat the dataframe to an one-dimensional array afterwards
- `interpret_prediction`: get the index of the one hot encoded prediction and return the label of the predicted index and the maximum of the prediction
- `define_variables`: set up the neural_net, the scaler and the data
- `calc_data_row`: build the data row based on the selected features
- `decision_emit`: If of one gesture is at least 5 times predicted, we emit the corresponding event. After every emitted event, we make a pause to prevent sending the same event multiple times.
- `emit_event`: send events to the server

`application.py` script opens the server for the slideshow and activates the tracking. We define the variables (neural net, scaler, data) at the beginning.
While application is running, we get the current data row with the processed pose and interpolate every two frames.
We extract data chunks of size 30. 
When we achieve a Dataframe with at least 30 frames, we flatten the data, transform them and use the neural net to make a prediction on these data.
If the prediction has a confidence > 0.7 or it is predicting idle, we append the prediction to our predictions array.
Afterwards we call the method `decision_emit`, that decides whether or not to emit an event. If it emitted an event, we clear the predictions. The last step is to remove the first frame from our dataframe.

## Performance score / Test mode
___
In `log_emitted_events_to_csv.py` we loop through the interpolated data with a chunk size of 15. We flatten and transform the data and predict a gesture for each chunk. After 4 predictions, the gesture event gets emitted.

The helpers for the test mode are in `preparation_and_prediciton.py` and work rather similar to the live mode:
- `flatten`: Flatten the dataframe to an one-dimensional array afterwards
- `interpolate`: Method to interpolate the data to 30 fps
- `interpret_prediction`: get the index of the one hot encoded prediction and return the label of the predicted index and the maximum of the prediction
- `define_variables`: set up the neural_net as well as the scaler
- `get_calc_data`: Take all frames and return the whole dataset with calulated features
- `get_decision`: If of one gesture is at least 4 times predicted, we emit the corresponding event. After every predicted event, we pause the prediction to prevent predicting the same event multiple times.

We first read in the csv containing the frames, define our neural net and scaler, and then we calculate our features and interpolate the data to 30fps. Then we make a prediction for every interpolated frame by advancing one frame at a time. This time we do not have a limit set for the confidence, every predicted event is also logged. After collecting all predictions, we need to downsample back to the original frequency. We do this by first extracting our predictions and generating a Dataframe with only idle. Then we pick the original timestamp that is closest to the prediction timestamp and set the prediction for this timestamp. Lastly our predictions are saved to a csv file.

## Utils
___
All data filenames are stored in the csv files `filenames.csv` and `filenames_test_mode.csv` to facilitate the import. 

In `utils` we created an enum for the gesture.

To help interpreting the predictions of our model, we wrote a method, located in `helpers.py`, which returns the index of the predicted one hot vector. 

### import data
In `import_data.py` all methods to import the data and methods for the data exploration can be found:
- `get_all_data`: imports all data and returns a dataframe with all data
- `get_all_data_by_gesture`: returns a dataframe with all data of the selected gesture
- `get_all_test_mode_data`: returns a dataframe with all data used for test mode
- `csv_file_to_dataframe`: reads one csv file and returns a dataframe  


The following methods are used for the data exploration:
- `get_all_data_by_gesture_and_body_part`: returns a dataframe with all data of the selected gesture and body part (e.g.: DOWN and nose)
- `get_gesture_length`: returns an integer array with the length of each gesture 
- `get_start_and_stop_time`: this method is used in get_gesture_length => we iterate over the frames and select the switch from idle to gesture (start) and from gesture to idle (stop)
- `get_mean_gesture_plot`: returns a plot with the mean gesture for the selected body part => iterates over all gestures and saves the gesture frames, afterwards it iterates over the longest gesture and selects the corresponding frames from all gestures and calculates the mean

### metrics
To calculate the metrics for a neural net, we created a class called `Metrics`. To instantiate this class, one needs the prediction, the ground truth as well as the interpreted prediction and the interpreted ground truth.
The method `print_metrics` returns the following:
- overall accuracy
- visualization for true / false positive / negative for each gesture
- confusion matrix
- accuracy, precision, recall and f1 score for each gesture
- macro precision
- macro recall
- macro f1 score

### preprocessing
The preprocessing consists of these steps:
1. get the data with the selected features
2. `get_gesture_chunks`: extract the gesture chunks from the data: loop through the data and check for changes between idle and a gesture, if change is detected extract the gesture as a chunk
3. `get_idle_as_array`: As we removed the gesture chunks from the data, this method returns arrays of continuous idle movements 
4. `get_idle_chunk`: extract idle chunks from all idle data
5. `get_final_data`: define X and Y (with and without ground truth) and flatten the data
6. shuffle all data
7. split the data in train (60%), validate (20%) and test (20%)
8. shuffle the train, validate and test data 
9. save the preprocessed data

## Hyperparameter optimization
___
The steps of the hyperparameter optimization:
1. Generate dataset
2. Iterations 
3. Learning rate (alpha) 
4. Activation functions: Sigmoid, ReLu, Tanh 
5. Lambda 
6. Feature Scaling
7. Number of layers and neurons
8. Input features:
    1. Chunk size
    2. How many features?
    3. "Special" features: subtracting, adding, multiplied
9. Final models:
    1. Live mode: 
        - Layers size: [570, 76, 38, 11]
        - Chunk size: 30
        - Activation function: tanh
        - iterations: 600
        - alpha: 0.0004
        - lambda: 0.0005
        - input features: subtracted features based on data exploration
    2. Test mode:
        - Layers size: [285, 76, 38, 4]
        - Chunk size: 15
        - Activation function: tanh
        - iterations: 600
        - alpha: 0.0004
        - lambda: 0.0005
        - input features: subtracted features based on data exploration


# Process videos

### `live_video_feed.py`: Example for live pose detection - a good way to test if MediaPipe installation is working on your machine!

Have a look at `live_video_feed.py`. It demonstrates how you can use MediaPipe to get the coordinates of human body keypoints, either directly from a webcam or from a video:

    python live_video_feed.py

### `video_to_csv.py`: Extract data from video

If you'd like to extract the keypoint coordinates from a video file and save the data for later processing (i.e., to train your model), have a look at `video_to_csv.py`: the script receives a video and writes the keypoint coordinates to a CSV file. This script is somewhat an extension of "live_video_feed.py".

    python video_to_csv.py

### `data_examples.ipynb`: Examples for what you can do with the data

You've already labeled your videos with the annotation tool "ELAN" and have extracted the keypoint coordinates with `video_to_csv.py`. What now? You need to somehow combine these data so your training routine can use it to train your machine learning models. In this notebook you will find examples how you can leverage pandas to do exactly that: read the CSV files and the exported files from ELAN to combine data and labels.

To open the Jupyter Notebook `data_examples.ipynb`:

  1. run a jupyter server
  2. open the notebook

Make sure to understand anything that happens in that notebook! It contains techniques you most certainly will require for your own implementations.
