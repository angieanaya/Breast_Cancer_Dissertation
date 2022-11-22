'''
@File   :   breastCancerClassification.py
@Date   :   04/11/2022
@Author :   María de los Ángeles Contreras Anaya
@Version:   1.0
@Desc:   Program that uses a CNN to classify lesion ins CESM images as benign, malignant or normal.
'''

import tensorflow as tf
import os
import cv2
import numpy as np
from datetime import datetime
from matplotlib import pyplot as plt
from keras.utils.vis_utils import plot_model
from tensorflow.keras.metrics import Precision, Recall
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout
from tensorflow.keras.models import load_model

def image_by_pathology():
    """
    Creates and image with 3 images, each one for each pathology
    """
    f, ax = plt.subplots(ncols=3, figsize=(10,10))
    benign = cv2.imread(os.path.join("Classification","Benign", "P100_L_CM_CC.jpg"))
    malignant = cv2.imread(os.path.join("Classification","Malignant", "P315_L_CM_MLO.jpg"))
    normal = cv2.imread(os.path.join("Classification","Normal", "P325_L_CM_MLO.jpg"))
    ax[0].imshow(normal)
    ax[0].set_title("Normal")
    ax[1].imshow(benign)
    ax[1].set_title("Benign")
    ax[2].imshow(malignant)
    ax[2].set_title("Malignant")
    f.tight_layout()
    plt.savefig('CESMPathology.png')

def plot_labels():
    """
    Plots three images with their assigned labels
    """
    fig, ax = plt.subplots(ncols=3, figsize=(20,20))
    fig.set_size_inches(20, 12)
    for idx, img in enumerate(batch[0][:3]):
        ax[idx].imshow(img.astype(int))
        ax[idx].title.set_text(batch[1][idx])
    plt.savefig('LabeledData.png')

def split_data():
    """
    Splits data into training, validation and testing sets

    Returns:
        train: Set of images for training the model
        val: Set of images for validation durint the training process of the model
        test: Set of images for evaluating the performance of the model
    """
    train_size = int(len(data)*.7)
    val_size = int(len(data)*.2)
    test_size = int(len(data)*.1)
    train = data.take(train_size)
    val = data.skip(train_size).take(val_size)
    test = data.skip(train_size+val_size).take(test_size)
    return train, val, test

def CNN(width, height, channels):
    """
    Defines the convolutional neural network that will be used for classification

    Returns:
        train: Set of images for training the model
        val: Set of images for validation durint the training process of the model
        test: Set of images for evaluating the performance of the model
    """
    # Implement convolutional neural network
    model = Sequential()
    # number of filters, size of filters, stride
    model.add(Conv2D(16, (3,3), 1, activation='relu', input_shape=(width, height, channels)))
    model.add(MaxPooling2D())
    model.add(Conv2D(32, (3,3), 1, activation='relu'))
    model.add(MaxPooling2D())
    model.add(Conv2D(32, (3,3), 1, activation='relu'))
    model.add(MaxPooling2D())
    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dense(3, activation='softmax'))
    model.compile("adam", loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=["accuracy"])
    plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)
    return model

def graph_loss(training):
    """
    Plot training and validation loss of the model
    """
    fig = plt.figure()
    plt.plot(training.history['loss'], color='teal', label='loss')
    plt.plot(training.history['val_loss'], color='orange', label='val_loss')
    fig.suptitle('Loss', fontsize=20)
    plt.legend(loc="upper right")
    plt.show()
    plt.savefig('loss_graph.png')

def graph_accuracy(training):
    """
    Plot training and validation accuracy of the model
    """
    fig = plt.figure()
    plt.plot(training.history['accuracy'], color='teal', label='accuracy')
    plt.plot(training.history['val_accuracy'], color='orange', label='val_accuracy')
    fig.suptitle('Accuracy', fontsize=20)
    plt.legend(loc="upper left")
    plt.show()
    plt.savefig('acc_graph.png')

def evaluate_model():
    """
    Evaluate model with testing split and display recall, precision and F1 scores
    """
    precision = Precision()
    recall = Recall()
    y_pred = []
    for batch in test.as_numpy_iterator():
        X, y = batch
        yhat = y_pred.predict(X)
        pred = np.max(yhat, axis=1)
        precision.update_state(y,pred)
        recall.update_state(y,pred)
    precision = precision.result().numpy()
    print("Precision = " + str(round(precision*100, 2)) + "%")
    recall = recall.result().numpy()
    print("Recall = " + str(round(recall*100, 2)) + "%")
    f1 = 2*((precision*recall)/(precision+recall))
    print("F1 = " + str(round(f1*100, 2)) + "%")

def predict_pre_trained_model(testing_img):
    """
    Plot training and validation accuracy of the model
    
    Parameters:
        testing_img(string): name of the image to classify
    """
    img = cv2.imread(testing_img) # read image
    test_image = tf.image.resize(img, (256,256)) # resize image
    diagnosis = bcc.predict(np.expand_dims(test_image/255, 0)) #scale image and expand dimension in axis=0
    max_val = max(diagnosis[0]) # get max_val since diagnosis is a probability vector
    index = np.where(diagnosis[0] == max_val) 
    # according to the index of max_val print the corresponding message
    if(index[0][0] == 0):
        dx="Diagnosis: Benign"
    elif(index[0][0]== 1):
        dx="Diagnosis: Malignant"
    else:
        dx="Diagnosis: Normal"
    # plot image
    plt.figure(figsize=(20, 15))
    plt.subplot(231)
    plt.title(dx)
    plt.imshow(test_image[:, :,0], cmap='gray')
    plt.show()

# load and label data
data = tf.keras.utils.image_dataset_from_directory(
    "Classification", #directory name
    class_names=["Benign", "Malignant", "Normal"])
# scale data
data = data.map(lambda x,y: (x/255, y))
# access pipeline
iterator = data.as_numpy_iterator()
batch = iterator.next() # access pipeline
plot_labels()
train, val, test = split_data()
model = CNN(256, 256, 3)
# train model
log_dir="logs/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3),
        tf.keras.callbacks.ModelCheckpoint(filepath='breastCancerClassifier.h5', monitor='val_loss', save_best_only=True),
        tf.keras.callbacks.TensorBoard(log_dir=log_dir)]

training = model.fit(train, epochs=20, validation_data=val, callbacks=callbacks)
graph_loss(training)
graph_accuracy(training)
evaluate_model()

bcc = load_model('breastCancerClassifier.h5') # load pre-trained model
predict_pre_trained_model('test/P9_L_CM_MLO.jpg') # classify a single image