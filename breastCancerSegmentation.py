'''
@File   :   breastCancerSegmentation.py
@Date   :   04/11/2022
@Author :   María de los Ángeles Contreras Anaya
@Version:   1.0
@Desc:   Program that trains a U-Net for breast cancer segmentation
'''

import tensorflow as tf 
import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
import random
from sklearn.model_selection import train_test_split
from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, concatenate, Conv2DTranspose, Dropout


def loadData(root_folder):
    """
    Loads images and their corresponding masks using cv2   
    
    Parameters:
        root_folder(string): string of the name(path) of the folder that contains images and masks

    Returns:
        images_data: List that contains the data of all images
        masks_data: List that contains the data of all masks
    """
    images = next(os.walk(root_folder))[1]
    images = sorted(images)
    for image_id in enumerate(images):
        if(os.path.isfile(os.path.join(root_folder,image_id[1], "masks", image_id[1] + ".png"))):
            img = cv2.imread(os.path.join(root_folder,image_id[1], image_id[1] + ".jpg")) #read image as RGB
            mask = cv2.imread(os.path.join(root_folder,image_id[1], "masks", image_id[1] + ".png"), 0) #read mask as greyscale
            #resize images and masks to 256x256
            img = cv2.resize(img, (256,256))
            mask = cv2.resize(mask, (256,256))
            #append the images to its corresponding array
            images_data.append(np.array(img))
            masks_data.append(np.array(mask))
    return images_data, masks_data

def printDataInformation(images_data, masks_data):
    """
    Displays information about the data loaded from images and masks to make sure shapes and values are good to go.
    
    Parameters:
        images_data: List that contains the data of all images
        masks_data: List that contains the data of all masks
    """
    print("Image data shape is: ", np.shape(images_data))
    print("Mask data shape is: ", np.shape(masks_data))
    print("Max pixel value in image is: ", np.max(images_data))
    print("Labels in the mask are : ", np.unique(masks_data))

def displayImages(images_data, masks_data, num):
    """
    Displays image and the mask over the image to see if the data was loaded correctly.
    
    Parameters:
        images_data(list): Data of loaded images
        masks_data(list): Data of the corresponding masks
        num(int): Random number between 1 - 484 that corresponds to a particular image

    """
    plt.figure(figsize=(16, 8))
    # display original image
    plt.subplot(241)
    plt.imshow(images_data[num], cmap="gray")
    # display mask over original image 
    plt.subplot(242)
    plt.imshow(images_data[num], cmap="gray")
    plt.imshow(masks_data[num,:,:,0], cmap="gray", alpha=0.3)

def UNet(img_height, img_width, img_channels):
    """
    Defines the U-Net model for semantic segmentation
    
    Parameters:
        img_height(int): height of the images that it receives as input
        img_width(int): width of the images that it receives as input
        img_channels(int): number of channels of the images that it receives as input

    """
    inputs = Input((img_height, img_width, img_channels))
    
    #Contraction path
    convL1 = Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(inputs)
    convL1 = Dropout(0.1)(convL1)
    convL1 = Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(convL1)
    poolingL1 = MaxPooling2D((2, 2))(convL1)
    
    convL2 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(poolingL1)
    convL2 = Dropout(0.1)(convL2)
    convL2 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(convL2)
    poolingL2 = MaxPooling2D((2, 2))(convL2)
     
    convL3 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(poolingL2)
    convL3 = Dropout(0.2)(convL3)
    convL3 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(convL3)
    poolingL3 = MaxPooling2D((2, 2))(convL3)
     
    convL4 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(poolingL3)
    convL4 = Dropout(0.2)(convL4)
    convL4 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(convL4)
    poolingL4 = MaxPooling2D(pool_size=(2, 2))(convL4)
     
    convL5 = Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(poolingL4)
    convL5 = Dropout(0.3)(convL5)
    convL5 = Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(convL5)
    
    #Expansive path 
    T1 = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(convL5)
    T1 = concatenate([T1, convL4])
    convL6 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(T1)
    convL6 = Dropout(0.2)(convL6)
    convL6 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(convL6)
     
    T2 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(convL6)
    T2 = concatenate([T2, convL3])
    convL7 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(T2)
    convL7 = Dropout(0.2)(convL7)
    convL7 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(convL7)
     
    T3 = Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(convL7)
    T3 = concatenate([T3, convL2])
    convL8 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(T3)
    convL8 = Dropout(0.1)(convL8)
    convL8 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(convL8)
     
    T4 = Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same')(convL8)
    T4 = concatenate([T4, convL1], axis=3)
    convL9 = Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(T4)
    convL9 = Dropout(0.1)(convL9)
    convL9 = Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(convL9)
     
    outputs = Conv2D(1, (1, 1), activation='sigmoid')(convL9)
     
    model = Model(inputs=[inputs], outputs=[outputs])
    model.compile(optimizer='adam', loss='binary_crossentropy')

    return model

def plotLoss(training):
    """
    Plots the training and validation loss at each epoch
    
    Parameters:
        training: data of the training process for the model

    """
    loss = training.history['loss']
    val_loss = training.history['val_loss']
    epochs = range(1, len(loss) + 1)
    plt.plot(epochs, loss, 'y', label='Training loss')
    plt.plot(epochs, val_loss, 'r', label='Validation loss')
    plt.title('Training and validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

def dice(y_pred, y_test):
    """
    Calculates dice score of the ground thruth image and the prediction    
    
    Parameters:
        y_pred: data of the prediction generated by the model
        y_test: data of the ground truth

    """
    intersection = np.logical_and(y_pred, y_test)
    dice_score = 2. * intersection.sum() / (y_pred.sum() + y_test.sum())
    return dice_score

def IoU(y_pred, y_test):
    """
    Calculates IoU of the ground thruth image and the prediction    
    
    Parameters:
        y_pred: data of the prediction generated by the model
        y_test: data of the ground truth

    """
    iou = tf.keras.metrics.IoU(num_classes=2, target_class_ids=[0,1])
    iou.update_state(y_pred, y_test)
    return iou

def display_results(random):
    """
    Prints the results of the prediction of one image.  
    
    Parameters:
        random(int):random number between 1 and the length of the test split

    """
    test_img = X_test[random]
    test_img_input=np.expand_dims(test_img, 0)
    ground_truth=y_test[random]
    ground_truth_thresholded = (ground_truth > 0.5).astype(np.uint8)
    prediction = (model.predict(test_img_input)[0,:,:,0] > 0.5)

    iou_score = IoU(ground_truth_thresholded, prediction) # IoU per image
    print("IoU score = " + str(round(iou_score.result().numpy()*100, 2)) + "%")

    dice_score = dice(ground_truth_thresholded, prediction) # dice per image
    print("Dice score = " + str(round(dice_score, 2)) + "%")

    # Plot image, label, prediction and label over prediction 
    plt.figure(figsize=(16, 8))
    plt.subplot(241)
    plt.title('Testing Image')
    plt.imshow(test_img[:, :,0], cmap='gray')
    plt.subplot(242)
    plt.title('Testing Label')
    plt.imshow(ground_truth[:,:,0], cmap='gray')
    plt.subplot(243)
    plt.title('Prediction')
    plt.imshow(prediction, cmap='gray')
    plt.subplot(244)
    plt.title('Testing label over prediction')
    plt.imshow(ground_truth[:,:,0], cmap='gray')
    plt.imshow(prediction, cmap='gray', alpha=0.7)
    plt.show()


def predict_pre_trained_model(model, image_filename):
    """
    Evaluates a new image with the pre-trained model and displays the results 
    
    Parameters:
        model: instance of the pre-trained model
        image_filename(String): name of the image to segment
    """
    test_img = cv2.imread(image_filename) # read image
    # pre-process image
    resized_img = cv2.resize(test_img, (256,256))
    test_img_input=np.expand_dims(resized_img, axis=0)/ 255.0
    prediction = (model.predict(test_img_input)[0,:,:,0] > 0.5)

    # plot prediction
    plt.figure(figsize=(16, 8))
    plt.subplot(231)
    plt.title('Testing Image')
    plt.imshow(resized_img[:, :,0], cmap='gray')
    plt.subplot(232)
    plt.title('Prediction')
    plt.imshow(prediction, cmap='gray')
    plt.subplot(233)
    plt.title('Prediction over Testing Image')
    plt.imshow(resized_img[:, :,0], cmap='gray')
    plt.imshow(prediction, cmap='gray', alpha=0.3)
    plt.show()

images_data, masks_data = loadData("Segmentation")
masks_data = np.expand_dims(masks_data,3) # add one dimension to masks
images_data = np.array(images_data)/255 # scale images
masks_data = masks_data /255 # scale masks
X_train, X_test, y_train, y_test = train_test_split(images_data, masks_data, test_size=0.2) # split data
model = UNet(images_data.shape[1], images_data.shape[2], images_data.shape[3]) # define U-Net model
callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5),
        tf.keras.callbacks.ModelCheckpoint(filepath='breastCancerDetection.h5', monitor='val_loss', save_best_only=True),
        tf.keras.callbacks.TensorBoard(log_dir='logs')]
training = model.fit(X_train, y_train, validation_split=0.1, batch_size=16, epochs=30, callbacks=callbacks) # train model
# Evaluate the model
y_pred=model.predict(X_test)
y_pred_thresholded = (y_pred > 0.5).astype(np.uint8)
y_test_thresholded = (y_test > 0.5).astype(np.uint8)
dice_score = dice(y_pred_thresholded, y_test_thresholded)
iou_score = IoU(y_pred_thresholded, y_test_thresholded)
print("Dice score = " + str(round(dice_score*100, 2)) + "%")
print("IoU score = " + str(round(iou_score.result().numpy()*100, 2)) + "%")
display_results(random.randint(0, len(X_test))) # visualize predictions from testing split

bcs = tf.keras.models.load_model('breastCancerDetection4.h5') # load pre-trained model
predict_pre_trained_model(model, "testing_imgs/P97_L_CM_MLO.jpg") # make predictions with pre-trained model