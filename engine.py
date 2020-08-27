#tutorial from https://www.youtube.com/watch?v=lUI6VMj43PE

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.layers import Conv2D, MaxPool2D
from tensorflow.keras.optimizers import Adam

import scipy.stats as stats


#loading the dataset file
file = open("WISDM_ar_v1.1/WISDM_ar_v1.1_raw.txt")
lines = file.readlines()

processedList = []
balanced_data = pd.DataFrame()
model = Sequential()
scaled_x = pd.DataFrame(data = X, columns = ["x","y", "z"])


Fs = 20 #frames per second
#=====================================
# ploting functions
def plot_activity(activity, data):
    fig, (ax0, ax1, ax2) = plt.subplots(nrows =3, figsize=(15,7), sharex=True)
    plot_axis(ax0, data["time"], data["x"], "X-Axis")
    plot_axis(ax1, data["time"], data["y"], "Y-Axis")
    plot_axis(ax2, data["time"], data["z"], "Z-Axis")
    plt.subplots_adjust(hspace = 0.2)
    fig.suptitle(activity)
    plt.subplots_adjust(top = 0.90)
    plt.show()


def plot_axis(ax, x, y ,title):
    ax.plot(x, y, "g")
    ax.set_title(title)
    ax.xaxis.set_visible(False)
    ax.set_ylim([min(y) - np.std(y), max(y) + np.std(y)])
    ax.set_xlim([min(x), max(x)])
    ax.grid(True)


def plotThem(activities, data):
    for activity in activities:
        data_for_plot = data[(data["activity"] == activity)] [:Fs *10]
        plot_activity(activity, data_for_plot)










#================================

#preprocessing
# =============================================================================
# for index, fileline in enumerate(lines):
#     #Extracting the data from the file
#     try:
#         line = fileline.split(",")
#         last = line[5].split(";")[0]
#         if last == "":
#             break
#         temp = [line[0], line[1], line[2], line[3], line[4], last]
#         processedList.append(temp)
#     except:
#         print("Error with line", index)
# =============================================================================

def preprocessing():
    for index, fileline in enumerate(lines):
        #Extracting the data from the file
        try:
            line = fileline.split(";")[0].split(",")
            if line[5] == "":
                break
            temp = line[0:6]
            processedList.append(temp)
        except:
            print("Error with line", index)


#second part
#balancing data
def balanceData(balanced_data):
    columns = ["user", "activity", "time", "x", "y", "z"]
    data = pd.DataFrame(data = processedList, columns = columns)


    #Changing the data type of x,y and z values from strings to floats
    data["x"] = data["x"].astype("float")
    data["y"] = data["y"].astype("float")
    data["z"] = data["z"].astype("float")

    #plotting the data for each activity
    # =============================================================================
    # activities = data["activity"].value_counts().index
    # plotThem(activities, data)
    # =============================================================================


    #removing timestamp and userid on dataset
    df = data.drop(["user","time"], axis=1).copy()
# =============================================================================
# print(df.head()) #looking ath the dataframe
# =============================================================================

# =============================================================================
# print(data["activity"].value_counts() ) #Gets information about the dataframe sizes
# =============================================================================

    #Balancing the datasets to avoid baises by creating dataframes for each activity
    walking = df[df["activity"] == "Walking"].head(48395).copy()
    jogging = df[df["activity"] == "Jogging"].head(48395).copy()
    upstairs = df[df["activity"] == "Upstairs"].head(48395).copy()
    downstairs = df[df["activity"] == "Downstairs"].head(48395).copy()
    sitting = df[df["activity"] == "Sitting"].head(48395).copy()
    standing = df[df["activity"] == "Standing"].head(48395).copy()

    #combining
    balanced_data = balanced_data.append([walking, jogging, upstairs, downstairs, sitting, standing])
    # =============================================================================
    # print(balanced_data["activity"].value_counts() )
    # =============================================================================

    #Making the activity a label
    label = LabelEncoder()
    balanced_data["label"] = label.fit_transform(balanced_data["activity"])
    return balanced_data

#checking if the encoding worked
# =============================================================================
# print(balanced_data.head())
# print(label.classes_)
# =============================================================================

#standardizing data
def standardizeData(scaled_x, balanced_data):
    X = balanced_data[["x","y","z"]]
    Y = balanced_data["label"]

    #standardizing x
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    scaled_x["label"] = Y.values

    print(scaled_x) #looking at the scaled dataframe
    return (scaled_x)





#Frame preparations
def prepareFrames(frame_size = Fs *4, hop_size = Fs*2, N_FEATURES = 3):
# =============================================================================
#     frame_size = Fs * 4 #getting data for 4 seconds (80 samples)
#     hop_size = Fs *2 #length of clip
# =============================================================================

# =============================================================================
#     #getting frames
#         N_FEATURES = 3
# =============================================================================

    frames = []
    labels = []
    for i in range(0, len(scaled_x) - frame_size, hop_size):
        xTemp = scaled_x["x"].values[i: i + frame_size]
        yTemp = scaled_x["y"].values[i: i + frame_size]
        zTemp = scaled_x["z"].values[i: i + frame_size]

        #retrieving the most used label in segment
        label = stats.mode(scaled_x["label"][i: i + frame_size])[0][0]
        frames.append([xTemp, yTemp ,zTemp])
        labels.append(label)

#Bring the segments into a better shape
    frames = np.asarray(frames).reshape(-1, frame_size, N_FEATURES)
    labels = np.asarray(labels)
    return frames, labels

# =============================================================================
# print("frames")
# print(frames.shape)
#
# print("labels")
# print(labels.shape)
# =============================================================================






#training
def reshapeData(frames, labels):
    x_train, x_test, y_train, y_test = train_test_split(frames, labels, test_size = 0.2, random_state = 0, stratify = labels)
    print(x_train.shape)
    print(x_test.shape)


    #reshaping training data into 3d because cnn can't work with 2d data
    x_train = x_train.reshape(5806, 80, 3, 1)
    x_test = x_test.reshape(1452, 80, 3, 1)
    return x_train, x_test, y_train, y_test



#2d CNN model
def buildModel(model, x_train, x_test, y_train, y_test):
    model.add( Conv2D(16, (2,2), activation = "relu", input_shape = x_train[0].shape) )
    model.add(Dropout(0.1))

    model.add(Conv2D (32, (2,2), activation = "relu"))
    model.add(Dropout(0.2))

    model.add(Flatten())

    model.add(Dense(64, activation = "relu"))
    model.add(Dropout(0.5))

    model.add(Dense(6, activation = "softmax"))


    model.compile(optimizer = Adam(learning_rate = 0.001), loss = "sparse_categorical_crossentropy", metrics = ["accuracy"])
    history = model.fit(x_train, y_train, epochs = 10, validation_data = (x_test, y_test), verbose =1)
    return history

def saveModel(model):
    model.save_weights("model weights.h5")





#Running the program
preprocessing()
balanced_data = balanceData(balanced_data)
scaled_x = standardizeData(scaled_x, balanced_data)
frames, labels = prepareFrames()
x_train, x_test, y_train, y_test = reshapeData(frames, labels)
history = buildModel(model, x_train, x_test, y_train, y_test)
saveModel(model)

