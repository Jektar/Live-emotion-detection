import tensorflow as tf
import tensorboard as tb
from PIL import Image, ImageOps
from tensorflow.keras.layers import Dense, Conv2D, Dropout, Flatten, MaxPooling2D, Activation, BatchNormalization
import os, PIL
import numpy as np

emotions = ['angry', 'disgusted', 'fearful', 'happy', 'sad', 'surprised', 'neutral']
emotionsDict = {'angry': 0, 'disgusted': 1, 'fearful': 2, 'happy': 3, 'sad': 4, 'surprised': 5, 'neutral': 6}

def formatData(image):
    image = image.resize((48, 48))
    image = ImageOps.grayscale(image)
    image = np.asarray(image)
    image = image.reshape(48, 48, 1)
    return image

def getImagesFromPath(path):
    r = []
    for file in os.listdir(path):
        image = PIL.Image.open((path + '/' + file), mode='r')
        r.append(formatData(image))

    return np.array(r)

def randomizeData(dataX, dataY):
    #randomize the data
    import random
    seed = random.random()
    random.seed(seed)
    random.shuffle(dataX)
    random.seed(seed)
    random.shuffle(dataY)
    return dataX, dataY

def loadData():
    # Get test and training data
    testPath = 'archive/test'
    trainPath = 'archive/train'

    testDataX = []
    testDataY = []
    trainDataX = []
    trainDataY = []

    for emotion in emotions:
        print('Loading ' + emotion + ' data...')
        for image in getImagesFromPath(testPath + '/' + emotion):
            testDataX.append(image)
            yVals = np.array([0 for i in range(7)])
            yVals[emotionsDict[emotion]] = 1
            testDataY.append(yVals)

        for image in getImagesFromPath(trainPath + '/' + emotion):
            trainDataX.append(image)
            yVals = np.array([0 for i in range(7)])
            yVals[emotionsDict[emotion]] = 1
            trainDataY.append(yVals)

    testDataX = np.array(testDataX)
    testDataY = np.array(testDataY)
    trainDataX = np.array(trainDataX)
    trainDataY = np.array(trainDataY)
    return randomizeData(trainDataX, trainDataY), randomizeData(testDataX, testDataY)

def creatModel():
    #create the model
    #Is this really a good model?
    model = tf.keras.Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', input_shape=(48, 48, 1)))
    model.add(BatchNormalization())

    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform'))
    model.add(BatchNormalization())

    model.add(Flatten())
    model.add(Dropout(0.5))
    model.add(Dense(64, activation='relu', kernel_initializer='he_uniform'))
    model.add(Dense(7, activation='softmax'))
    return model

def getClassWeights(trainDataY):
    #get the class weights
    classWeights = {0: 1, 1: 1, 2: 1, 3: 1, 4: 1, 5: 1, 6: 1}
    numberOfInstances = [0, 0, 0, 0, 0, 0, 0]
    for y in trainDataY:
        for i in range(7):
            numberOfInstances += y[i]

    for i in range(7):
        classWeights[i] = (len(trainDataY)/2) / numberOfInstances[i]
    return classWeights

def learn(epochs=10):

    (trainX, trainY), (testX, testY) = loadData()
    classWeights = getClassWeights(trainY)
    model = creatModel()

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    history = model.fit(trainX, trainY, epochs=epochs, validation_data=(testX, testY), class_weight=classWeights)
    model.save('emotion_detector.h5')

    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(2)
    axs[0].plot(history.history['val_loss'], label='val_loss')
    axs[0].plot(history.history['loss'], label='loss')
    axs[1].plot(history.history['val_accuracy'])
    axs[1].plot(history.history['accuracy'])


    loss, acc = model.evaluate(testX, testY, verbose=2)
    print("Loss: " + str(loss))
    print("Accuracy: " + str(acc))
    plt.show()

def loadModel():
    model = tf.keras.models.load_model('emotion_detector.h5')
    return model

def predict(image):
    model = loadModel()
    image = formatData(image)
    image = image.reshape(1, 48, 48, 1)
    prediction = model.predict(image)
    return prediction

if __name__ == "__main__":
    learn(10)