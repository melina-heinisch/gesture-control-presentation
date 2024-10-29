import os
import argparse
import pandas as pd
import pathlib
import numpy as np

from preparation_and_prediction import flatten, interpret_prediction, get_decision, define_variables, get_calc_data, interpolate
here = pathlib.Path(__file__).parent

parser = argparse.ArgumentParser()
parser.add_argument("--input_csv", help="CSV file containing the video transcription from OpenPose", required=True)
parser.add_argument("--output_csv_name", help="output CSV file containing the events", default="emitted_events.csv")

args = parser.parse_known_args()[0]

input_path = args.input_csv

output_directory, input_csv_filename = os.path.split(args.input_csv)
output_path = "%s/%s" % (output_directory, args.output_csv_name)

# Read in csv
# Define neural net and scaler
# Calculate our features
# Interpolate the frames
# Make variables for temporary results, predictions per frame and chunk size 
frames = pd.read_csv(input_path)

neural_net, scaler = define_variables(here)
data = get_calc_data(frames)
data = interpolate(data)
temp = pd.DataFrame(index=data.index, columns=["events"]) #Make new Dataframe with timestamp index
predictions = []
chunk_size = 15

# We start at Frame 15, then take the current plus 14 previous frames
# We flatten and scale this chunk
# Then we make a prediction for the chunk and interpret it
# Each prediction gets added to an array, which is then used to make a descion on wheter or not an event is occuring
# In specific cases we reset the predictions, and the result of the decision process is added to our temp results
for i in range(chunk_size, len(data)):
  start = i-chunk_size
  chunk = data.iloc[start:i]
  chunk = flatten(chunk, chunk_size)
  chunk = scaler.transform(chunk)
  predict_raw = neural_net.predict(chunk)
  #print(np.around(predict_raw, 2))
  prediction, confidence = interpret_prediction(predict_raw)
  #print(f"Predict {prediction} with {confidence}")
  predictions.append(prediction) 
  value,reset = get_decision(predictions)
  if reset:
     predictions = []
  temp.iloc[i] = value

# If there are any cells with no value, add idle there
# We now get all predictions that are not idle
# Then we create a dataframe with the initial timestamps as index and with an "events" column and fill this with idle
# For each non-idle prediction, we get the initial index that is closest to the interpolated index and set this event for the index
# Lastly we save the results with the correct index
temp.loc[temp["events"].isnull()] = "idle"

predictions = temp[temp["events"] !="idle"]

result = pd.DataFrame(index=frames["timestamp"], columns=["events"]) #create new empty dataframe
result.loc[result["events"].isnull()] = "idle" #set all events to idle
result = result.set_index(pd.to_timedelta(frames["timestamp"], unit="ms")) #set timestamp as index to compare later

for idx, row in predictions.iterrows():
  new_index = result.index.get_loc(idx, method='nearest') #find the initial index that is closest to our prediction index
  result.iloc[new_index] = row["events"]

result = result.set_index(frames["timestamp"])
result.to_csv(output_path, index=True) # since "timestamp" is the index, it will be saved also
print("events exported to %s" % output_path)
