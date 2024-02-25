import tensorflow as tf
import matplotlib.pyplot as plt
import random
from PIL import Image
import numpy as np
import os

import tensorflow as tf
import os
import numpy as np
from PIL import Image

'''
@inproceedings{liu2016ssd,
  Author = {Liu, Wei and Anguelov, Dragomir and Erhan, Dumitru and Szegedy, Christian and Reed, Scott and Fu, Cheng-Yang and Berg, Alexander C.},
  Booktitle = {European Conference on Computer Vision (ECCV)},
  Title = {SSD: Single Shot MultiBox Detector},
  Year = {2016}
}
'''

class VGG16(tf.keras.Model):
# Defining the class that contains the VGG16 backbone
    def __init__(self):
    # Procedure to initialise the VGG16 layers
        super().__init__()
        # Calls the parent class to ensure it is executed first
        self.vgg16 = tf.keras.applications.VGG16(weights='imagenet', include_top=False, input_shape=(300, 300, 3))
        # Initialises the VGG16 model with weights from a dataset called imagenet

        self.layer4_3_5_3 = tf.keras.models.Model(inputs=self.vgg16.input, outputs=[self.vgg16.get_layer('block4_conv3').output, self.vgg16.get_layer('block5_conv3').output])
        # Declares the inputs as self.vgg16 and the output as the requied layers

    def call(self, tensor):
    # Function to pass the input image tensor through the VGG16 layers
        layer4_3, layer5_3 = self.layer4_3_5_3(tensor)

        # Passes the tensor through the layers and stores the outputs in respective variables
        return layer4_3, layer5_3

class SSDLayers(tf.keras.Model):
# Defining the class that contains the SSD additional layers
    def __init__(self):
    # Procedure to initialise the SSD additional layers
        super().__init__()
        # Calls the parent class to ensure it is executed first
        self.layer6 = tf.keras.Sequential([
        # Keras Sequential layer to hold all layers in block 6
            tf.keras.layers.Conv2D(filters=1024, kernel_size=3, padding='same', activation='relu'),
            tf.keras.layers.BatchNormalization(),
        ])
        self.layer7 = tf.keras.Sequential([
        # Keras Sequential layer to hold all layers in block 7
            tf.keras.layers.Conv2D(filters=1024, kernel_size=1, padding='same', activation='relu'),
            tf.keras.layers.BatchNormalization(),
        ])
        self.layer8 = tf.keras.Sequential([
         # Keras Sequential layer to hold all layers in block 8
            tf.keras.layers.Conv2D(filters=256, kernel_size=1, padding='same', activation='relu'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Conv2D(filters=512, kernel_size=3, strides=2, padding='same', activation='relu'),
            tf.keras.layers.BatchNormalization(),
        ])
        self.layer9 = tf.keras.Sequential([
        # Keras Sequential layer to hold all layers in block 9
            tf.keras.layers.Conv2D(filters=128, kernel_size=1, padding='same', activation='relu'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Conv2D(filters=256, kernel_size=3, strides=2, padding='same', activation='relu'),
            tf.keras.layers.BatchNormalization(),
        ])
        self.layer10 = tf.keras.Sequential([
        # Keras Sequential layer to hold all layers in block 10
            tf.keras.layers.Conv2D(filters=128, kernel_size=1, padding='same', activation='relu'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Conv2D(filters=256, kernel_size=3, strides=2, padding='same', activation='relu'),
            tf.keras.layers.BatchNormalization(),
        ])
        self.layer11 = tf.keras.Sequential([
        # Keras Sequential layer to hold all layers in block 11
            tf.keras.layers.Conv2D(filters=128, kernel_size=1, padding='same', activation='relu'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Conv2D(filters=256, kernel_size=3, activation='relu'),
            tf.keras.layers.BatchNormalization(),
        ])

    def call(self, tensor):
    # Function to pass the VGG16 ouput tensor through the SSD additional layers
        layer6 = self.layer6(tensor)
        layer7 = self.layer7(layer6)
        layer8 = self.layer8(layer7)
        layer9 = self.layer9(layer8)
        layer10 = self.layer10(layer9)
        layer11 = self.layer11(layer10)
        # Passes the tensor through the layers and stores the outputs in respective variables
        return layer7, layer8, layer9, layer10, layer11

class SSDPrediction(tf.keras.Model):
# Class to take the feature maps and turn them into predictions
    def __init__(self):
    # Procedure to initialise the SSD prediction layers
        super().__init__()
        # Calls the parent class to ensure it is executed first
        self.numClasses = 1
        NUMPREDICTIONS = [4, 8, 12, 12, 8, 4]
        # Initialises constants that are used to determine how many filters for each feature map prediction

        self.fmap37Class = tf.keras.layers.Conv2D(filters=NUMPREDICTIONS[0], kernel_size=3, padding='same', activation='sigmoid')
        self.fmap18Class = tf.keras.layers.Conv2D(filters=NUMPREDICTIONS[1], kernel_size=3, padding='same', activation='sigmoid')
        self.fmap9Class = tf.keras.layers.Conv2D(filters=NUMPREDICTIONS[2], kernel_size=3, padding='same', activation='sigmoid')
        self.fmap5Class = tf.keras.layers.Conv2D(filters=NUMPREDICTIONS[3], kernel_size=3, padding='same', activation='sigmoid')
        self.fmap3Class = tf.keras.layers.Conv2D(filters=NUMPREDICTIONS[4], kernel_size=3, padding='same', activation='sigmoid')
        self.fmap1Class = tf.keras.layers.Conv2D(filters=NUMPREDICTIONS[5], kernel_size=3, padding='same', activation='sigmoid')
        # Initialises the convolutional layers that will be used to predict where the faces are

        self.fmap37Loc = tf.keras.layers.Conv2D(filters=(NUMPREDICTIONS[0] * 4), kernel_size=3, padding='same')
        self.fmap18Loc = tf.keras.layers.Conv2D(filters=(NUMPREDICTIONS[1] * 4), kernel_size=3, padding='same')
        self.fmap9Loc = tf.keras.layers.Conv2D(filters=(NUMPREDICTIONS[2] * 4), kernel_size=3, padding='same')
        self.fmap5Loc = tf.keras.layers.Conv2D(filters=(NUMPREDICTIONS[3] * 4), kernel_size=3, padding='same')
        self.fmap3Loc = tf.keras.layers.Conv2D(filters=(NUMPREDICTIONS[4] * 4), kernel_size=3, padding='same')
        self.fmap1Loc = tf.keras.layers.Conv2D(filters=(NUMPREDICTIONS[5] * 4), kernel_size=3, padding='same')
        # Initialises the convolutional layers that will be used to predict the bounding box offsets

    def call(self, layer4_3, layer7, layer8, layer9, layer10, layer11):
    # Function to pass the feature map outputs through prediction layers
        fmap37Class = self.fmap37Class(layer4_3)
        fmap18Class = self.fmap18Class(layer7)
        fmap9Class = self.fmap9Class(layer8)
        fmap5Class = self.fmap5Class(layer9)
        fmap3Class = self.fmap3Class(layer10)
        fmap1Class = self.fmap1Class(layer11)
        # Passes each feature map through its respective layer

        fmap37Loc = self.fmap37Loc(layer4_3)
        fmap18Loc = self.fmap18Loc(layer7)
        fmap9Loc = self.fmap9Loc(layer8)
        fmap5Loc = self.fmap5Loc(layer9)
        fmap3Loc = self.fmap3Loc(layer10)
        fmap1Loc = self.fmap1Loc(layer11)
        # Passes each feature map through its respective layer

        featureMapsLoc = [tf.reshape(fmap, (tf.shape(fmap)[0], -1, 4)) for fmap in [fmap37Loc, fmap18Loc, fmap9Loc, fmap5Loc, fmap3Loc, fmap1Loc]]
        # Reshapes each prediction into a way thats easier to understand
        featureMapsClass = [tf.reshape(fmap, (tf.shape(fmap)[0], -1, self.numClasses)) for fmap in [fmap37Class, fmap18Class, fmap9Class, fmap5Class, fmap3Class, fmap1Class]]
        # Reshapes each prediction into a way thats easier to understand

        localisation = tf.concat(featureMapsLoc, axis=1)
        # Concatenates the predictions into one variable
        classifcation = tf.concat(featureMapsClass, axis=1)
        # Concatenates the predictions into one variable

        return tf.concat([localisation, classifcation], axis=2)

class SSD(tf.keras.Model):
# Class to take the image input and output the final prediction
    def __init__(self):
    # Procedure to instantiate objects from the VGG16, SSDLayers and SSDPrediction classes
        super().__init__()
        # Calls the parent class to ensure it is executed first

        self.vgg16 = VGG16()
        # Instantiates a VGG16 object from the class
        self.ssdlayers = SSDLayers()
        # Instantiates a SSDLayers object from the class
        self.ssdprediction = SSDPrediction()
        # Instantiates a SSDPrediction object from the class

    def call(self, tensor):
    # Function to pass the input image through the VGG16, SSDLayers and SSDPrediction classes
        layer4_3, layer5_3 = self.vgg16(tensor)
        layer7, layer8, layer9, layer10, layer11 = self.ssdlayers(layer5_3)
        prediction = self.ssdprediction(layer4_3, layer7, layer8, layer9, layer10, layer11)
        # Passes the image tensor through the different layers and stores their outputs in respective variables 
        return prediction
    
def anchorBoxGeneration(featureMaps, sizes, aspectRatios):
# Function to generate the anchor boxes
    anchorBoxes = []
    # Initalises a list to store all anchor boxes
    for i, feature in enumerate(featureMaps):
    # Iterates through each feature map
        for y in range(feature[0]):
        # Iterates across the x-axis for each feature map
            for x in range(feature[1]):
            # Iterates across the y-axis for each feature map
                for size in sizes[i]:
                # Iterates through the different sizes for that feature map
                    for ratio in aspectRatios[i]:
                    # Iterates through the ratios for that feature map
                        w = size * np.sqrt(ratio)
                        h = size / np.sqrt(ratio)
                        # Calculates the width and height of the anchor boxes

                        x1 = x / feature[0]
                        y1 = y / feature[1]
                        # Normalises the values of x1 and y1 

                        if 1 - x1 < w:
                            w = 1 - x1
                        if 1 - y1 < h:
                            h = 1 - y1
                        # Ensures the anchor boxes do not go outside of the image

                        anchorBoxes.append([y1, x1, y1+h, x1+w])
                        # Appends the anchor box to the list of all anchor boxes 

    return anchorBoxes

def preprocessImage(path):
    # Function to preprocess the image
    image = Image.open(path)
    # Opens the image at the specified path
    image = image.resize((300, 300))
    # Resizing the image to 300x300 pixels
    image = image.convert('RGB')
    # Ensures image is in the RGB format
    image = np.array(image, dtype=np.float32)
    # Converting the image into a tensor 
    image = image / 255.0
    # Normalising the image [0-1]
    image = tf.reshape(image, (1, 300, 300, 3))
    # Reshapes the image to fit the model input size
    return image

FEATUREMAPS = [(37, 37), (18, 18), (9, 9), (5, 5), (3, 3), (1, 1)]
ASPECTRATIOS = [[1.0, 0.8], [1.0, 0.8], [1.0, 0.8, 0.6], [1.0, 0.8, 0.6], [1.0, 0.8], [1.0, 0.8]]
SIZES = [[0.1, 0.2], [0.1, 0.2, 0.4, 0.6], [0.1, 0.2, 0.4, 0.6], [0.1, 0.2, 0.4, 0.6], [0.1, 0.2, 0.4, 0.6], [0.1, 0.2]]
# Initialises the hyperparamters that generate the anchor boxes
anchorBoxes = anchorBoxGeneration(FEATUREMAPS, SIZES, ASPECTRATIOS)
# Calls function to generate anchor boxes

anchorBoxesNMS = tf.convert_to_tensor(anchorBoxes, dtype=np.float32)
anchorBoxesNMS = tf.reshape(anchorBoxesNMS, (len(anchorBoxes), 4))
# Converts anchorBoxes into the correct format to go into the NMS algorithm

model = SSD()
# Creates an instance of the model
model.load_weights('faceDetection/cp.ckpt')
# Loads model weights from the file saved during training

def faceDetectionModel(image):
# Function to pass images through the face detection model
    try:
        Image.open(image).verify() 
        preprocessedImg = preprocessImage(image)
        # Preprocesses the image into required model form
        prediction = model.predict(preprocessedImg, verbose=0)
        # Puts the image through the model to get the desired output
        prediction = tf.cast(prediction[:, :, -1:], tf.float32)
        # Removes the bounding box offsets
        
        bboxIndex, bboxScores = tf.image.non_max_suppression_with_scores(
            boxes=anchorBoxesNMS, 
            scores=tf.reshape(prediction, (len(anchorBoxes))), 
            max_output_size=50, 
            iou_threshold=0.1,
            score_threshold=0.3
        )
        # Carries out non-max supression on the image

        bboxList = [anchorBoxes[index] for index in (bboxIndex.numpy())]
        # Creates the list of bounding box coordinates given the indexes
        
        if len(bboxList) == 1:
        # Validates that there is at least one box to crop
            bboxTensor = tf.convert_to_tensor(bboxList, dtype=np.float32)
            # Converts the bboxList into the required format to be cropped

            boxIndices = [0 for i in range(len(bboxList))]
            # Calculates the indices of each cropped images

            croppedImages = tf.image.crop_and_resize(
                image=preprocessedImg,
                boxes=bboxTensor,
                box_indices=boxIndices,
                crop_size=[224,224],
            )
            # Crops out each face detected in the image and resizes it to the required model format

            return croppedImages
        else:
            return []
        
    except Exception as e:
        print(image + ' provides exception ' + e)
        return []

    

