# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name：     convolutional stacked AE
   Description :
   Author :       yabing
   date：          2021/3/2
-------------------------------------------------

"""

# Import keras and used Layers (Tensorflow 2.1)
import keras
from keras.layers import Conv2D, MaxPooling2D, Conv2DTranspose, UpSampling2D, Dropout
from keras.models import Sequential, load_model
from keras.layers import LayerNormalization

# numpy: math stuff, pathlib: loading and working with paths, PIL: image toolbox, matplotlib: plotting images and stuff
import numpy as np
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt
from keras.utils import plot_model
from sklearn import metrics
import os
import shutil
import re
import scipy.io as sio
import tensorflow as tf

path_images_train_AD = Path(r'/home/yabing/KIT/Hiwi/ITIV_anomaly_detection/UCSD_Anomaly_Dataset.v1p2/UCSDped1/')  # set to path to images with UCSDped1
# base paths to Train and Test folders
path_UCSD_Anomaly_ped1_Train = path_images_train_AD.resolve().joinpath(r"Train")
path_UCSD_Anomaly_ped1_Test = path_images_train_AD.resolve().joinpath(r"Test")

# extract folder paths for each recording
Train_Data_folders = np.sort([path for path in path_UCSD_Anomaly_ped1_Train.glob("*") if "Train" in path.name])
Test_Data_folders = np.sort([path for path in path_UCSD_Anomaly_ped1_Test.glob("*") if "Test" in path.name])

# Convolutional AutoEncoder
def build_model():
    model = Sequential()

    # Convolutional layers
    model.add(Conv2D(512, (11, 11), strides=4, padding="same", activation='relu', input_shape=(208, 208, 10)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(256, (5, 5), padding="same", activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(128, (3, 3), padding="same", activation='relu'))

    # Deconxolutional layers
    model.add(Conv2DTranspose(128, (3, 3), padding="same", activation='relu'))
    model.add(UpSampling2D(size=(2, 2)))
    model.add(Conv2DTranspose(256, (3, 3), padding="same", activation='relu'))
    model.add(UpSampling2D(size=(2, 2)))
    model.add(Conv2DTranspose(512, (5, 5), padding="same", activation='relu'))
    model.add(Dropout(0.2))
    model.add(Conv2DTranspose(10, (11, 11), strides=4, padding="same", activation='sigmoid'))

    print(model.summary())

    adam = keras.optimizers.Adam(lr=0.001, decay=0.95, epsilon=1e-6)
    model.compile(loss='mse', optimizer=adam, metrics=['accuracy'])
    return model


# Function for loading data
def load_train_data(source_path, number_frames_per_batch=10, jump=1, train=True):
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
    number_frames = len(frame_files)
    list_frames = list()
    list_batches = list()

    # Get all 200 Frames into list
    for path in frame_files:
        frame = Image.open(path)

        # resize Frame to (230, 230) format and norm to (0,1)
        if train:
            frame = np.array(frame.resize((230, 230))) / 255
            a, b = np.random.randint(0, 22, 2)
            frame_shifted = frame[a:a+208, b:b+208]

            # append all frames to a list
            list_frames.append(frame_shifted)
        else:
            frame = np.array(frame.resize((208, 208))) / 255
            list_frames.append(frame)

    # now we have to create the input/target batches by appending number_frames_per_batch frames with jump
    number_batches = number_frames - number_frames_per_batch * jump

    # for each batch that can be created
    for i in range(number_batches):
        batch = np.zeros((208, 208, 10))

        # for each of the number_frames_per_batch frames
        for j in range(number_frames_per_batch):
            batch[:, :, j] = list_frames[i + j * jump]

        # transform batch from list to array and append to output list
        list_batches.append(batch)

    # return list with all batches
    return list_batches


def get_gt_files(test_data_path, save_dir):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    folders = []
    for folder in os.listdir(test_data_path):
        if 'gt' in folder:
            folders.append(folder)
    for folder in folders:
        dst_path = os.path.join(save_dir, folder)
        if not os.path.exists(dst_path):
            os.makedirs(dst_path)
            src_path = os.path.join(test_data_path, folder)
            for file in os.listdir(src_path):
                shutil.move(f'{src_path}/{file}', dst_path)
            os.rmdir(src_path)

def get_anomaly_score(test_data_path):
    # label_file = ""
    # for file in test_data_path.glob("*"):
    #     if ".m" in file.name:
    #         label_file = file
    # labels = label_file.read_text(encoding='utf-8')

    for file in os.listdir(test_data_path):
        if '.m' in file:
            label_file = file
    labels = ''
    with open(f'{str(test_data_path)}/{label_file[2:]}', 'r') as f:
        labels = f.readlines()

    # split the string
    label1 = []
    for label in labels:
        label = label.split(';')
        label1.append(label[0])
    label2 = []
    for label in label1:
        label = label.split('=')
        label2.append(label[1])
    label2 = label2[1:]
    label3 = []
    for label in label2:
        label3.append(list(re.findall(r"\d*:\d*", label)))
    gt_dc = {}
    for i in range(len(label3)):
        labels1 = label3[i]
        gt = np.zeros(200)
        for label in labels1:
            start_ind = int(label.split(':')[0])
            end_ind = int(label.split(':')[1])
            gt[start_ind: end_ind] = 1
        gt_dc[str(i+1)] = gt
    print(gt_dc)

def load_test_data(test_data_path, number_frames_per_batch=10, jump=1):
    # labels = [label for label in labels label.splitlines()]
    test_files = [file for file in os.listdir(test_data_path) if 'Test' in file]
    # for i in range(len(test_files)):
    #     label_dic[test_files[i]] = labels[i+1]


    # frame_files = np.sort([path for path in source_path.glob("*") if ".tif" in path.name])
    # number_frames = len(frame_files)
    #
    # list_frames = list()
    # list_batches = list()
    #
    # # Gett all 300 Frames into list
    # for path in frame_files:
    #     frame = Image.open(path)
    #
    #     # resize Frame to (232,232) format and norm to (0,1)
    #     frame = np.array(frame.resize((208, 208))) / 255
    #
    #     # append all frames to a list
    #     list_frames.append(frame)
    #
    # # now we have to create the input/target batches by appending number_frames_per_batch frames with jump
    #
    # number_batches = number_frames - number_frames_per_batch * jump
    #
    # # for each batch that can be created
    # for i in range(number_batches):
    #     batch = np.zeros((208, 208, 10))
    #
    #     # for each of the number_frames_per_batch frames
    #     for j in range(number_frames_per_batch):
    #         batch[:, :, j] = list_frames[i + j * jump]
    #
    #     # transform batch from list to array and append to output list
    #     list_batches.append(batch)

def train_plot(model, train_data, test_data, model_name):
    # fit model - example (see keras documentation for validation split and stuff)
    my_callbacks = [keras.callbacks.EarlyStopping(monitor="val_loss", patience=10, mode="min"),
                    keras.callbacks.TensorBoard(log_dir='./logs')]
    # keras.callbacks.ModelCheckpoint(filepath='./results/model.{epoch:02d}-{val_loss:.2f}.h5',
    #                                                 save_best_only=True)
    history = model.fit(train_data, train_data, validation_data=(test_data, test_data),
                        batch_size=2, epochs=50, verbose=1, callbacks=my_callbacks)

    # evaluate the model
    _, train_acc = model.evaluate(train_data, train_data, verbose=1)
    _, test_acc = model.evaluate(test_data, test_data, verbose=1)
    print('Train: %.3f, Test: %.3f' % (train_acc, test_acc))

    model_dir = f'./results/{model_name}'
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    # accuracy curve
    # plot history
    # plt.plot(history.history['accuracy'])
    # plt.plot(history.history['val_accuracy'])
    # plt.title('Model accuracy')
    # plt.ylabel('Accuracy')
    # plt.xlabel('Epoch')
    # plt.legend(['Train', 'Test'], loc='upper left')
    # plt.savefig('./results/Model_accuracy.png')
    # plt.show()

    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.savefig(f'{model_dir}/Model_loss.png')
    plt.show()

    # save the model as h5 file
    plot_model(model, show_shapes=True, to_file=f'{model_dir}/{model_name}.png')
    model.save(f'{model_dir}/{model_name}.h5')


def get_auc(model_path, test_data_path=Test_Data_folders[0]):
    model = keras.models.load_model(model_path)
    test_data = load_train_data(test_data_path, train=False)
    test_data = np.asarray(test_data)
    # auc_ls = []
    # for i in range(len(data)):
    # test_data_1 = np.reshape(test_data[0], [1, 208, 208, 10])
    # pred = model.predict(test_data_1)
    pred = model.predict(test_data)

    frames_per_batch = test_data.shape[3]
    batch_num = test_data.shape[0]
    score_frame_ls = []
    score_batch_ls = []
    # compute anomaly score per frame
    for i in range(batch_num):
        for j in range(frames_per_batch):
            score_frame = np.sum((pred[i,:,:,j] - test_data[i,:,:,j]) ** 2) / (208 * 208)
            score_frame_ls.append(score_frame)
        score_batch_ls.append(sum(score_frame_ls))



    # for i in range(frames_per_batch):
    #     pred = pred[:, :, :, i].reshape(208, 208, 1)
    #     data = test_data[:, :, :, i].reshape(208, 208, 1)
    #     anomaly_score = np.sum((pred - data) ** 2) / (208 * 208)
    #     score_ls.append(anomaly_score)
    # score_per_batch =
    # auc = metrics.roc_auc_score(data_1, pred)
    fpr, tpr, thresholds = metrics.roc_curve(test_data_1, pred, pos_label=1)
    auc = metrics.auc(fpr, tpr)
    print(auc)

    # plt.plot(fpr, tpr, label='ROC curve (area = %0.3f)' % auc)
    # plt.plot([0, 1], [0, 1], 'k--')  # random predictions curve
    # plt.xlim([0.0, 1.0])
    # plt.ylim([0.0, 1.0])
    # plt.xlabel('False Positive Rate or (1 - Specifity)')
    # plt.ylabel('True Positive Rate or (Sensitivity)')
    # plt.title('Receiver Operating Characteristic')
    # plt.legend(loc="lower right")

def predict(model_path):
    frames_per_batch = 10
    model = keras.models.load_model(model_path)
    data0 = load_train_data(Test_Data_folders[0], train=False)
    data1 = np.reshape(data0[0], [1, 208, 208, 10])
    pred = model.predict(data1)

    pred_dir = './results/predictions'
    # pred_dir = './results/predictions/Conv_stacked_AE_v2'
    if not os.path.exists(pred_dir):
        os.makedirs(pred_dir)

    for i in range(frames_per_batch):
        fig = plt.figure()
        image_array = pred[:, :, :, i].reshape(208, 208, 1)
        plt.imshow(image_array, cmap='viridis')
        # cmp: 'viridis', like depth map, gray: gray scaled
        plt.title(i)
        plt.show()
        fig.savefig(f'{pred_dir}/{str(i)}.png', dpi=fig.dpi)


def main():
    # build and train model
    model = build_model()

    # for i in range(1, len(Train_Data_folders)):
    #     list_extracts = load_train_data(Train_Data_folders[i], number_frames_per_batch=10, jump=1)
    #     train_arr_all = np.concatenate((train_arr, np.asarray(list_extracts)))
    #     train_arr = train_arr_all
    # np.save('./results/train.npy', train_arr_all)
    # print('train.npy is successfully saved')

    # np.random.shuffle(train_arr_all)
    # row_num = train_arr_all.shape[0]
    # auto_encoder_train = train_arr_all[0:0.8*row_num]
    # auto_encoder_val = train_arr_all[0.8*row_num:]

    model_name = 'Conv_stacked_AE_v3'

    '''load one training recording'''
    # list_extracts = load_train_data(Train_Data_folders[0], number_frames_per_batch=10, jump=1)
    # row_num = len(list_extracts)
    # auto_encoder_train = np.asarray(list_extracts[:int(0.8 * row_num)])
    # auto_encoder_val = np.asarray(list_extracts[int(0.8 * row_num):])
    # train_plot(model, auto_encoder_train, auto_encoder_val, model_name)

    '''load all training recordings and train iteratively'''
    for i in range(len(Train_Data_folders)):
        list_extracts = load_train_data(Train_Data_folders[i], number_frames_per_batch=10, jump=1)
        row_num = len(list_extracts)
        auto_encoder_train = np.asarray(list_extracts[:int(0.8*row_num)])
        auto_encoder_val = np.asarray(list_extracts[int(0.8*row_num):])
        train_plot(model, auto_encoder_train, auto_encoder_val, model_name)

    '''load all training recordings by choosing small fractions'''
    list_all_extracts = list()
    for recording_path in Train_Data_folders:
        list_extracts = load_train_data(recording_path, number_frames_per_batch=10, jump=1)
        list_all_extracts.append(list_extracts)

    split = 2
    times = len(list_all_extracts) // split
    for i in range(times + 1):
        if i == times:
            auto_encoder_input = np.asarray(list_all_extracts[split*i:])
            row_num = auto_encoder_input.shape[1]
            auto_encoder_train = np.reshape(auto_encoder_input[:, 0: int(0.8 * row_num), :, :], [-1, 208, 208, 10])
            auto_encoder_val = np.reshape(auto_encoder_input[:, int(0.8 * row_num):, :, :], [-1, 208, 208, 10])
            train_plot(model, auto_encoder_train, auto_encoder_val, model_name)
        else:
            auto_encoder_input = np.asarray(list_all_extracts[split*i:split*(i+1)])
            row_num = auto_encoder_input.shape[1]
            auto_encoder_train = np.reshape(auto_encoder_input[:, 0: int(0.8*row_num), :, :], [-1, 208, 208, 10])
            auto_encoder_val = np.reshape(auto_encoder_input[:, int(0.8*row_num):, :, :], [-1, 208, 208, 10])
            train_plot(model, auto_encoder_train, auto_encoder_val, model_name)


    # auto_encoder_input = np.expand_dims(auto_encoder_input, axis=4)
    # get anomaly score
    # get_anomaly_score(path_UCSD_Anomaly_ped1_Test)

    # get auc curve
    model_path = f'./results/models/{model_name}/{model_name}.h5'
    # get_auc(model_path)

    # make predictions
    # gt_dir = os.path.join('/home/yabing/KIT/Hiwi/ITIV_anomaly_detection/UCSD_Anomaly_Dataset.v1p2/UCSDped1', 'Test_gt')
    # get_gt_files(path_UCSD_Anomaly_ped1_Test, gt_dir)
    # load_test_data(path_UCSD_Anomaly_ped1_Test, number_frames_per_batch=10, jump=1)
    # predict(model_path)


def scratch():
    a, b = np.random.randint(0, 32, 2)
    print(a, b)
    arr = np.load('./results/train.npy')
    pass

if __name__ == '__main__':
    main()
    # scratch()
