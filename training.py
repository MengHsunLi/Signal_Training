from __future__ import print_function
import pandas as pd
import numpy as np
import math

ap_SSID = pd.Series(['AP1', 'AP2', 'AP3', 'AP4', 'AP5'])
ap_BSSID = pd.Series(['00:1b:2f:a8:e5:21', '00:1b:2f:a8:e5:22', '00:1b:2f:a8:e5:23', '00:1b:2f:a8:e5:24', '00:1b:2f:a8:e5:25'])
ap_isSecure = pd.Series([True, False, True, False, False])
ap_SignalStrength = pd.Series([-30, -30, -30, -30, -30])
loc = pd.Series(['Loc1', 'Loc2', 'Loc3', 'Loc4'])
device_wifi_first = pd.Series(['AP2', 'AP3', 'AP1', 'AP5'])
device_wifi_second = pd.Series(['AP3', 'AP5', 'AP4', 'AP4'])
device_wifi_third = pd.Series(['AP1', 'AP4', 'AP5', 'AP3'])
device_wifi_dist_first = pd.Series([0.9014, 0.5000, 0.5590, 0.5590])
device_wifi_dist_second = pd.Series([1.1180, 0.9014, 0.7071, 0.7071])
device_wifi_dist_third = pd.Series([1.3463, 1.5811, 1.5207, 1.5000])
#device_wifi_signal_first = pd.Series([0.8197, 0.9, 0.8882, 0.8882])
#device_wifi_signal_second = pd.Series([0.7764, 0.8197, 0.8586, 0.8586])
#device_wifi_signal_third = pd.Series([0.7307, 0.6838, 0.6959, 0.7])
def distToLost(freq, lossFactor, dist, floor):
    '''
        freq: Frequency in MHz
        lossFactor: -全開放環境:2.0~2.5 -半開放環境:2.5~3.0 -較封閉環境:3.0~3.5 -隧道環境:1.6~1.8
        dist: Distance in meters
        floor: Floors between AP and device
    '''
    loss = 20*np.log(freq)+10*lossFactor*np.log(dist)+6+3*(floor-1)
    return loss


device_loc = pd.Series(['Loc1', 'Loc2', 'Loc3', 'Loc4'])
location_x = pd.Series([0, 1, 0, 1])
location_y = pd.Series([0, 0, 1, 1])

ap = pd.DataFrame({'SSID':ap_SSID, 'BSSID':ap_BSSID, 'Secure':ap_isSecure})
# SSID        BSSID       Secure
#  AP1  00:1b:2f:a8:e5:21  True
#  AP2  00:1b:2f:a8:e5:22  False
#  AP3  00:1b:2f:a8:e5:23  True
#  AP4  00:1b:2f:a8:e5:24  False
#  AP5  00:1b:2f:a8:e5:25  True
location = pd.DataFrame({'Location':loc, 'Location_X':location_x, 'Location_Y':location_y})
# Location Location_X Location_Y
#   Loc1       0          0
#   Loc2       1          0
#   Loc3       0          1
#   Loc4       1          1
device_wifi_signal_first_loss = distToLost(2400, 3.0, device_wifi_dist_first, 1)
device_wifi_signal_second_loss = distToLost(2400, 3.0, device_wifi_dist_second, 1)
device_wifi_signal_third_loss = distToLost(2400, 3.0, device_wifi_dist_third, 1)

print("device_wifi_signal_first_loss:")
print(device_wifi_signal_first_loss)
print("device_wifi_signal_second_loss:")
print(device_wifi_signal_second_loss)
print("device_wifi_signal_third_loss:")
print(device_wifi_signal_third_loss)

device_wifi_signal_first_loss = 1/pow(10, np.log10((device_wifi_signal_first_loss)/10))
device_wifi_signal_second_loss = 1/pow(10, np.log10((device_wifi_signal_second_loss)/10))
device_wifi_signal_third_loss = 1/pow(10, np.log10((device_wifi_signal_third_loss)/10))

device_wifi_signal_first = 500*device_wifi_signal_first_loss
device_wifi_signal_second = 500*device_wifi_signal_second_loss
device_wifi_signal_third = 500*device_wifi_signal_third_loss


device = pd.DataFrame({'WIFI_SSID_First':device_wifi_first,'WIFI_Signal_First':device_wifi_signal_first,
                        'WIFI_SSID_Second':device_wifi_second, 'WIFI_Signal_Second':device_wifi_signal_second,
                        'WIFI_SSID_Third':device_wifi_third, 'WIFI_Signal_Third':device_wifi_signal_third,
                        'Location':device_loc})
# Location WIFI_SSID_First WIFI_SSID_Second WIFI_Signal_First WIFI_Signal_Second
#   Loc1         AP2             AP3             0.8197             0.7764
#   Loc2         AP3             AP5             0.9000             0.8197
#   Loc3         AP1             AP4             0.8882             0.8586
#   Loc4         AP5             AP4             0.8882             0.8586



device_wifi_bssid_first = []
device_wifi_isSecure_first = []
for item in device['WIFI_SSID_First']:
    wifi_mask = ap['SSID'].isin([item])
    device_wifi_bssid_first.append(ap[wifi_mask].iloc[0]['BSSID'])
    device_wifi_isSecure_first.append(ap[wifi_mask].iloc[0]['Secure'])
device_loc_x = []
device_loc_y = []
for item in device['Location']:
    loc_mask = location['Location'].isin([item])
    device_loc_x.append(location[loc_mask].iloc[0]['Location_X'])
    device_loc_y.append(location[loc_mask].iloc[0]['Location_Y'])

device.insert(loc=7, column='Location_Y', value=device_loc_y)
device.insert(loc=7, column='Location_X', value=device_loc_x)
device.insert(loc=0, column='WIFI_Secure_First', value=device_wifi_isSecure_first)
#device.insert(loc=0, column='WIFI_BSSID_First', value=device_wifi_bssid_first)

# Location  Location_X Location_Y WIFI_SSID_First WIFI_Signal_First     WIFI_BSSID     WIFI_Secure
#   Loc1        0          0            AP2            0.8197        00:1b:2f:a8:e5:22     False
#   Loc2        1          0            AP3            0.9000        00:1b:2f:a8:e5:23     True
#   Loc3        0          1            AP1            0.8882        00:1b:2f:a8:e5:21     True
#   Loc4        1          1            AP5            0.8882        00:1b:2f:a8:e5:25     False

device_wifi_one_hot_first = pd.get_dummies(device['WIFI_SSID_First'])
for item in ap_SSID:
    if item not in device_wifi_one_hot_first:
        device_wifi_one_hot_first[item] = [0, 0, 0, 0]
device_wifi_one_hot_first=device_wifi_one_hot_first*3
device_wifi_one_hot_second = pd.get_dummies(device['WIFI_SSID_Second'])
for item in ap_SSID:
    if item not in device_wifi_one_hot_second:
        device_wifi_one_hot_second[item] = [0, 0, 0, 0]
device_wifi_one_hot_second=device_wifi_one_hot_second*2
device_wifi_one_hot_third = pd.get_dummies(device['WIFI_SSID_Third'])
for item in ap_SSID:
    if item not in device_wifi_one_hot_third:
        device_wifi_one_hot_third[item] = [0, 0, 0, 0]
device_wifi_one_hot = device_wifi_one_hot_first + device_wifi_one_hot_second + device_wifi_one_hot_third
device = device.join(device_wifi_one_hot)

device_location_one_hot = pd.get_dummies(device['Location'])
device = device.join(device_location_one_hot)

newdevice = device
for i in range(499):
    newdevice = pd.concat([newdevice, device],axis=0, ignore_index=True)

noise_df = pd.DataFrame(np.random.random((2000,3)), columns=['Bias_First', 'Bias_Second', 'Bias_Third'])
noise_df*=10
noise_df_2 = pd.DataFrame(np.random.choice([-1, 1], size=(2000, 3), p=[0.5, 0.5]), columns=['Rate_First', 'Rate_Second', 'Rate_Third'])
noise_df = pd.concat([noise_df, noise_df_2], axis=1)
noise_df['WIFI_Signal_First']=noise_df['Bias_First'].multiply(noise_df['Rate_First'], axis=0)
noise_df['WIFI_Signal_Second']=noise_df['Bias_Second'].multiply(noise_df['Rate_Second'], axis=0)
noise_df['WIFI_Signal_Third']=noise_df['Bias_Third'].multiply(noise_df['Rate_Third'], axis=0)

for item in newdevice:
    if item in noise_df:
        newdevice[item]+=noise_df[item]


for item in newdevice['WIFI_Signal_First']:
    if item < 0:
        newdevice['WIFI_Signal_First'].item = 0.0



print("ap:")
print(ap)
print("\nlocation:")
print(location)
print("\ndevice:")
print(device)
print("\nnoise:")
print(noise_df)
print("\nnewdevice:")
print(newdevice)


#Training

import math

from matplotlib import cm
from matplotlib import gridspec
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from sklearn import metrics
import tensorflow as tf
from tensorflow.python.data import Dataset

tf.logging.set_verbosity(tf.logging.ERROR)
pd.options.display.max_rows = 10
pd.options.display.float_format = '{:.1f}'.format

newdevice = newdevice.reindex(np.random.permutation(newdevice.index))

def preprocess_features(device):
  """Prepares input features from Device data set.

  Args:
    device: A Pandas DataFrame expected to contain data
      from the Device data set.
  Returns:
    A DataFrame that contains the features to be used for the model, including
    synthetic features.
  """
  selected_features = device[
    ["Location",
     "WIFI_SSID_First", "WIFI_SSID_Second", "WIFI_SSID_Third",
     "WIFI_Signal_First", "WIFI_Signal_Second", "WIFI_Signal_Third",
     "AP1", "AP2", "AP3", "AP4", "AP5",
     "Loc1", "Loc2", "Loc3", "Loc4"
    ]]
  processed_features = selected_features.copy()
  return processed_features

def preprocess_targets(device):
  """Prepares target features (i.e., labels) from Device data set.

  Args:
    device: A Pandas DataFrame expected to contain data
      from the Device data set.
  Returns:
    A DataFrame that contains the target feature.
  """
  output_targets = pd.DataFrame()
  # Scale the target to be in units of thousands of dollars.
  output_targets["Location_X"] = (
    device["Location_X"])
  return output_targets

# Choose the first 1800 (out of 2000) examples for training.
training_examples = preprocess_features(newdevice.head(1800))
training_targets = preprocess_targets(newdevice.head(1800))
# Choose the last 200 (out of 2000) examples for validation.
validation_examples = preprocess_features(newdevice.tail(200))
validation_targets = preprocess_targets(newdevice.tail(200))

correlation_dataframe = training_examples.copy()
correlation_dataframe["target"] = training_targets["Location_X"]
correlation_dataframe.corr()

def construct_feature_columns(input_features):
  """Construct the TensorFlow Feature Columns.

  Args:
    input_features: The names of the numerical input features to use.
  Returns:
    A set of feature columns
  """
  return set([tf.feature_column.numeric_column(my_feature)
              for my_feature in input_features])

def my_input_fn(features, targets, batch_size=1, shuffle=True, num_epochs=None):
    """Trains a linear regression model.

    Args:
      features: pandas DataFrame of features
      targets: pandas DataFrame of targets
      batch_size: Size of batches to be passed to the model
      shuffle: True or False. Whether to shuffle the data.
      num_epochs: Number of epochs for which data should be repeated. None = repeat indefinitely
    Returns:
      Tuple of (features, labels) for next data batch
    """

    # Convert pandas data into a dict of np arrays.
    features = {key:np.array(value) for key,value in dict(features).items()}

    # Construct a dataset, and configure batching/repeating.
    ds = Dataset.from_tensor_slices((features,targets)) # warning: 2GB limit
    ds = ds.batch(batch_size).repeat(num_epochs)

    # Shuffle the data, if specified.
    if shuffle:
      ds = ds.shuffle(10000)

    # Return the next batch of data.
    features, labels = ds.make_one_shot_iterator().get_next()
    return features, labels

def train_model(
    learning_rate,
    steps,
    batch_size,
    training_examples,
    training_targets,
    validation_examples,
    validation_targets):
  """Trains a linear regression model.

  In addition to training, this function also prints training progress information,
  as well as a plot of the training and validation loss over time.

  Args:
    learning_rate: A `float`, the learning rate.
    steps: A non-zero `int`, the total number of training steps. A training step
      consists of a forward and backward pass using a single batch.
    batch_size: A non-zero `int`, the batch size.
    training_examples: A `DataFrame` containing one or more columns from
      `california_housing_dataframe` to use as input features for training.
    training_targets: A `DataFrame` containing exactly one column from
      `california_housing_dataframe` to use as target for training.
    validation_examples: A `DataFrame` containing one or more columns from
      `california_housing_dataframe` to use as input features for validation.
    validation_targets: A `DataFrame` containing exactly one column from
      `california_housing_dataframe` to use as target for validation.

  Returns:
    A `LinearRegressor` object trained on the training data.
  """

  periods = 50
  steps_per_period = steps / periods

  # Create a linear regressor object.
  my_optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
  my_optimizer = tf.contrib.estimator.clip_gradients_by_norm(my_optimizer, 5.0)
  linear_regressor = tf.estimator.LinearRegressor(
      feature_columns=construct_feature_columns(training_examples),
      optimizer=my_optimizer
  )

  # Create input functions.
  training_input_fn = lambda: my_input_fn(training_examples,
                                          training_targets["Location_X"],
                                          batch_size=batch_size)
  predict_training_input_fn = lambda: my_input_fn(training_examples,
                                                  training_targets["Location_X"],
                                                  num_epochs=1,
                                                  shuffle=False)
  predict_validation_input_fn = lambda: my_input_fn(validation_examples,
                                                    validation_targets["Location_X"],
                                                    num_epochs=1,
                                                    shuffle=False)

  # Train the model, but do so inside a loop so that we can periodically assess
  # loss metrics.
  print("Training model...")
  print("RMSE (on training data):")
  training_rmse = []
  validation_rmse = []
  for period in range (0, periods):
    # Train the model, starting from the prior state.
    linear_regressor.train(
        input_fn=training_input_fn,
        steps=steps_per_period,
    )
    # Take a break and compute predictions.
    training_predictions = linear_regressor.predict(input_fn=predict_training_input_fn)
    training_predictions = np.array([item['predictions'][0] for item in training_predictions])

    validation_predictions = linear_regressor.predict(input_fn=predict_validation_input_fn)
    validation_predictions = np.array([item['predictions'][0] for item in validation_predictions])

    # Compute training and validation loss.
    training_root_mean_squared_error = math.sqrt(
        metrics.mean_squared_error(training_predictions, training_targets))
    validation_root_mean_squared_error = math.sqrt(
        metrics.mean_squared_error(validation_predictions, validation_targets))
    # Occasionally print the current loss.
    print("  period %02d : %0.2f" % (period, training_root_mean_squared_error))
    # Add the loss metrics from this period to our list.
    training_rmse.append(training_root_mean_squared_error)
    validation_rmse.append(validation_root_mean_squared_error)
  print("Model training finished.")


  # Output a graph of loss metrics over periods.
  plt.ylabel("RMSE")
  plt.xlabel("Periods")
  plt.title("Root Mean Squared Error vs. Periods")
  plt.tight_layout()
  plt.plot(training_rmse, label="training")
  plt.plot(validation_rmse, label="validation")
  plt.legend()

  return linear_regressor

minimal_features = [
  "WIFI_Signal_First",
  "WIFI_Signal_Second",
  "WIFI_Signal_Third",
  "AP1", "AP2", "AP3", "AP4", "AP5",
  "Loc1", "Loc2", "Loc3", "Loc4"
]

minimal_training_examples = training_examples[minimal_features]
minimal_validation_examples = validation_examples[minimal_features]

_ = train_model(
    learning_rate=0.0005,
    steps=500,
    batch_size=10,
    training_examples=minimal_training_examples,
    training_targets=training_targets,
    validation_examples=minimal_validation_examples,
    validation_targets=validation_targets)
