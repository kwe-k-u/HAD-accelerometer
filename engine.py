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


class HAS_engine:

    def __init__(self):
        #defining attributes
        self.processedList = []
        self.balanced_data = pd.DataFrame()
        self.model = Sequential()

        self.Fs = 20 #contain the number of frames per second
        self.frame_size = self.Fs * 4
        self.hop_size = self.Fs * 2
        self.numFeatures = 3

        self.fig = None
        self.data = None
        self.filelines = None


        self.x_train = None
        self.x_test = None
        self.y_train = None
        self.y_test = None

        self.scaled_x = None
        self.frames = None
        self.labels = None










    #ploting the data from the accelerometer datasets
    def plot_activity(self):
        self.fig, (x_ax, y_ax, z_ax) = plt.subplots(nrows = 3, figsize = (15,7), sharex=True)
        self.plot_axis(x_ax, self.data["time"], self.data["x"], "X-Axis")
        self.plot_axis(y_ax, self.data["time"], self.data["y"], "Y-Axis")
        self.plot_axis(z_ax, self.data["time"], self.data["z"], "Z-Axis")
        plt.subplots_adjust(hspace = 0.2)
        self.fig.suptitle(self.activity)
        plt.suptitle_adjust(top = 0.90)
        plt.show()


    def plot_axis(self, ax, x, y ,title):
        ax.plot(x, y, "g")
        ax.set_title(title)
        ax.xaxis.set_visible(False)
        ax.set_ylim([min(y) - np.std(y), max(y) + np.std(y)])
        ax.set_xlim([min(x), max(x)])
        ax.grid(True)



    def plotThem(self,activities, data):
        for activity in activities:
            data_for_plot = data[(data["activity"] == activity)] [:self.Fs *10]
            self.plot_activity(activity, data_for_plot)

    #reads the dataset file into the engine
    def loadData(self):
        file = open("WISDM_ar_v1.1/WISDM_ar_v1.1_raw.txt")
        self.filelines = file.readlines()

    #Preprocesses the datasets for learning
    def preprocessing(self):
        for index, fileLine in enumerate(self.filelines):
            try:
                line = fileLine.split(";")[0].split(",")
                if line[5] == "":
                    break
                self.processedList.append( line[0:6] )
            except:
                print("Error with line", index)


    #Balances the dataset
    def balancedData(self):
        columns = ["user", "activity", "time", "x", "y", "z"]
        self.data = pd.DataFrame( data = self.processedList, columns = columns)


        #Changing the data type of x,y and z values from strings to floats
        self.data["x"] = self.data["x"].astype("float")
        self.data["y"] = self.data["y"].astype("float")
        self.data["z"] = self.data["z"].astype("float")

        #removing timestand and userid from datasets
        tempFrame = self.data.drop(["user", "time"], axis =1).copy()

        #the number was chosen from the number smallest number of datasets per activity
        walking = tempFrame[ tempFrame["activity"] == "Walking"].head(48395).copy()
        jogging = tempFrame[ tempFrame["activity"] == "Jogging"].head(48395).copy()
        upstairs = tempFrame[ tempFrame["activity"] == "Upstairs"].head(48395).copy()
        downstairs = tempFrame[ tempFrame["activity"] == "Downstairs"].head(48395).copy()
        sitting = tempFrame[ tempFrame["activity"] == "Sitting"].head(48395).copy()
        standing = tempFrame[ tempFrame["activity"] == "Standing"].head(48395).copy()

        #combing the dataframes
        #todo this line was balanced_data = balanced_data ...
        self.balanced_data = self.balanced_data.append([walking, jogging, upstairs, downstairs, sitting, standing])

        #creating an activity label
        label = LabelEncoder()
        self.balanced_data["label"] = label.fit_transform( self.balanced_data["activity"])



    #standardizing the data
    def standardizeData(self):
        X = self.balanced_data[["x", "y", "z"]]
        Y = self.balanced_data["label"]
        self.scaled_x = pd.DataFrame(data = X, columns = ["x", "y", "z"])

        scale = StandardScaler()
        X = scale.fit_transform(X)

        self.scaled_x["label"] = Y.values

        print(self.scaled_x)


    #Preparing the frames for analysing
    def prepareFrames(self):
        framesTemp=[]
        labelsTemp = []



        for i in range(0, len(self.scaled_x) - self.frame_size, self.hop_size ):
            xtemp = self.scaled_x["x"].values[i: i + self.frame_size]
            ytemp = self.scaled_x["y"].values[i: i + self.frame_size]
            ztemp = self.scaled_x["z"].values[i: i + self.frame_size]

            tempLabel = stats.mode( self.scaled_x["label"][i: i + self.frame_size] )[0][0]
            framesTemp.append( [xtemp, ytemp, ztemp] )
            labelsTemp.append(tempLabel)

            self.frames = np.asarray(framesTemp).reshape(-1, self.frame_size, self.numFeatures)
            self.labels = np.asarray(labelsTemp)

    #Reshaping data
    def reshapeData(self):
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(self.frames, self.labels, test_size = 0.2, random_state = 0, stratify = self.labels)

        #reshaping training data into 3d because cnn can't work with 2d data
        self.x_train = self.x_train.reshape(5806,80,3,1)
        self.x_test = self.x_test.reshape(1452,80,3,1)



    #building the 3d CNN model
    def buildModel(self):
        self.model.add( Conv2D(16, (2,2), activation = "relu", input_shape = self.x_train[0].shape) )
        self.model.add(Dropout(0.1))

        self.model.add( Conv2D(32, (2,2), activation = "relu") )
        self.model.add(Dropout(0.2))

        self.model.add(Flatten())

        self.model.add(Dense(64, activation = "relu"))
        self.model.add(Dropout(0.5))

        self.model.add(Dense(6, activation = "softmax"))

        self.model.compile(optimizer = Adam(learning_rate = 0.001), loss = "sparse_categorical_crossentropy", metrics = ["accuracy"])
        self.history = self.model.fit( self.x_train, self.y_train, epochs = 10, validation_data = (self.x_test, self.y_test), verbose = 1)

    def saveModel(self):
        self.model.save_weights("Model weights.h5")



#running the program
engine = HAS_engine()

engine.loadData()
engine.preprocessing()
engine.balancedData()
engine.standardizeData()
engine.prepareFrames()
engine.reshapeData()
engine.buildModel()
engine.saveModel()