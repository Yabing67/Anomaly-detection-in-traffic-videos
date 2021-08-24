# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name：     Autoencoder
   Description :
   Author :       yabing
   date：          2021/2/22
-------------------------------------------------

"""
# Import keras and used Layers (Tensorflow 2.1)
import keras
from keras.layers import Conv2DTranspose, ConvLSTM2D, BatchNormalization, TimeDistributed, Conv2D
from keras.models import Sequential, load_model
from keras.layers import LayerNormalization

# numpy: math stuff, pathlib: loading and working with paths, PIL: image toolbox, matplotlib: plotting images and stuff
import numpy as np
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt

### Example Network ###
def function_getter():
    act = "relu"  # activation function for convolutional layers
    input_dimension = (None, 10, 232, 232, 1)  # (None, no_frames, height, width, channels) <- only grey scale

    seq = Sequential()

    # # # # # Encoding # # # # #
    seq.add(TimeDistributed(Conv2D(64, (11, 11), strides=4, padding="same", activation=act),
                            batch_input_shape=input_dimension))
    seq.add(TimeDistributed(Conv2D(128, (5, 5), strides=2, padding="same", activation=act)))

    # # # # # LSTM # # # # #
    seq.add(ConvLSTM2D(64, (3, 3), padding="same", return_sequences=True, activation="sigmoid"))
    seq.add(ConvLSTM2D(32, (3, 3), padding="same", return_sequences=True, activation="sigmoid"))
    seq.add(ConvLSTM2D(64, (3, 3), padding="same", return_sequences=True, activation="sigmoid"))

    # # # # # Decoding # # # # #
    seq.add(TimeDistributed(Conv2DTranspose(128, (5, 5), strides=2, padding="same", activation=act)))
    seq.add(TimeDistributed(Conv2DTranspose(64, (11, 11), strides=4, padding="same", activation=act)))
    seq.add(TimeDistributed(Conv2D(1, (11, 11), activation="sigmoid", padding="same")))

    print(seq.summary())

    seq.compile(loss='mse', optimizer=keras.optimizers.Adam(lr=1e-4, decay=1e-5, epsilon=1e-6))

    return seq


model = function_getter()

# Function for loading data
def load_data(source_path, number_frames_per_batch=10, jump=2):
    """
    inputs:
    source_path: string: path to folder with frame images
    number_frames: int: number of frames
    jump: int: number of frames between time steps

    output:
    list_batches: list with nummpy arrays of format [number_frames, format_image_x, format_image_y]
    """

    # load all image-paths in source_path
    frame_files = np.sort([path for path in source_path.glob("*") if ".tif" in path.name])
    # b = source_path.glob(".tif")
    # frame_files = np.sort([b])
    number_frames = len(frame_files)

    list_frames = list()
    list_batches = list()

    # Gett all 300 Frames into list
    for path in frame_files:
        frame = Image.open(path)

        # resize Frame to (232,232) format and norm to (0,1)
        frame = np.array(frame.resize((232, 232))) / 255

        # append all frames to a list
        list_frames.append(frame)

    # now we habe to create the input/target batches by appending number_frames_per_batch frames with jump

    number_batches = number_frames - number_frames_per_batch * jump

    # for each batch that can be created
    for i in range(number_batches):
        batch = list()

        # for each of the number_frames_per_batch frames
        for j in range(number_frames_per_batch):
            batch.append(list_frames[i + j * jump])

        # transform batch from list to array and append to output list
        batch = np.asarray(batch)
        list_batches.append(batch)

    # return list with all batches
    return list_batches


path_images_train_AD = Path(r'/home/yabing/KIT/Hiwi/ITIV_anomaly_detection/UCSD_Anomaly_Dataset.v1p2/UCSDped1/')  # set to path to images with UCSDped1
# base paths to Train and Test folders
path_UCSD_Anomaly_ped1_Train = path_images_train_AD.resolve().joinpath(r"Train")
path_UCSD_Anomaly_ped1_Test = path_images_train_AD.resolve().joinpath(r"Test")

# extract folder paths for each recording
Train_Data_folders = np.sort([path for path in path_UCSD_Anomaly_ped1_Train.glob("*") if "Train" in path.name])
Test_Data_folders = np.sort([path for path in path_UCSD_Anomaly_ped1_Test.glob("*") if "Test" in path.name])

# example loading for one recording
list_extracts = load_data(Train_Data_folders[0], number_frames_per_batch=10, jump=2)

# transform list to array
auto_encoder_input = np.asarray(list_extracts)

'''
Example to load all batches from all Training Recordings
list_all_extracts = list()

for recording_path in Train_Data_folders:
    list_extracts = load_data(Train_Data_folders[0],number_frames_per_batch = 10, jump = 2)

    list_all_extracts.append(list_extracts) 

auto_encoder_input = np.asarray(list_all_extracts)
'''
auto_encoder_input = np.expand_dims(auto_encoder_input, axis=4)

# fit model - example (see keras documentation for validation split and stuff)
model.fit(auto_encoder_input, auto_encoder_input, batch_size=1, epochs=10)
# save the model as h5 file
model.save('model.h5')