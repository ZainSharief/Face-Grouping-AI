import tensorflow as tf
from faceDetection import faceDetectionModel
import os
import random

def dataExtactor(annotationFilePath):
# Function to extract the image paths and celebIDs from the csv file
    imagePathList = []
    # A list to store all the image paths
    celebIDList = []
    # A list to store all the celeb IDs

    with open(annotationFilePath) as file:
        # Opens the annotation file
        for line in file:
        # Iterates through each line in the annotation file
            imagePath, celebID = line.strip().split(" ")
            # Seperates the line into imagePath and celebID
            imagePathList.append(imagePath)
            celebIDList.append(celebID)
            # Appends the image paths and celebIDs to their respective lists

    return imagePathList, celebIDList

def randomNegative(imagePathList, celebIDList, celebID):
# Function to generate random negative image
    
    negativeImgNum = random.randrange(0, len(imagePathList)-1)
    # Selects a random number that corresponds to an image

    while celebIDList[negativeImgNum] == celebID:
    # Validates that the people in the photos are different
        negativeImgNum = random.randrange(0, len(imagePathList)-1)
        # Selects a random number that corresponds to an image

    return imagePathList[negativeImgNum]

def tripletImage(imagePathList, celebIDList):
# Function to group each image into 3 - anchor, positive and negative
   
    model = faceNet()
    # Creates an instance of the model
    model.load_weights('faceRecognitionWeights/cp.ckpt')
    # Loads in the model weights
    THRESHOLD = 0.75
    # Defines the model threshold for a positive sample

    totalPos = 0
    totalNeg = 0
    correctPos = 0
    correctNeg = 0
    # Initialises variables to calculate the model accuracy
   
    visited = []
    # Initialises a validation list to ensure we are not repeating a person

    for celebID in celebIDList:
    # Iterates through each celebID
        if celebID not in visited:
        # Validation to check if we have already selected this individual
                visited.append(celebID)
                # Adds the individual to the visited list so they cannot be selected again
                indices = [i for i, x in enumerate(celebIDList) if x == celebID]
                # Finds all the indices of images that contain that person 

                indices = indices[:10]
                # Limits the number of images per person to 20 due to computational resource limitations
                
                for i in range(0, len(indices)-1, 2):  
                # Iterates through the images and skips any remainder
                    anchor = os.path.join('img_celeba', imagePathList[indices[i]])
                    positive = os.path.join('img_celeba', imagePathList[indices[i+1]])
                    # Define the anchor image and positive image 

                    negative = os.path.join('img_celeba', randomNegative(imagePathList, celebIDList, celebID))
                    # Picks a random image to be the negative sample

                    anchor = faceDetectionModel(anchor)
                    if len(anchor) != 0:
                        positive = faceDetectionModel(positive)
                        if len(positive) != 0:
                            negative = faceDetectionModel(negative)
                            if len(negative) != 0:
                            # Validation to make sure each image has 1 detected face in it
                                
                                anchor = model.predict(anchor, verbose=0)
                                positive = model.predict(positive, verbose=0)
                                negative = model.predict(negative, verbose=0)
                                # Passes all three images through the model

                                posDistance = tf.reduce_sum(tf.square(anchor - positive), axis=-1)
                                negDistance = tf.reduce_sum(tf.square(anchor - negative), axis=-1)  
                                # Calculates the squared euclidean distance              

                                print(posDistance)
                                print(negDistance)
                                # Prints out the squared euclidean distance values
                                
                                if posDistance <= THRESHOLD:
                                # Validates the positive distance is less than the threshold
                                    totalPos += 1
                                    correctPos += 1
                                else:
                                    totalPos += 1
                                # Calculates the positive accuracy 

                                if negDistance > THRESHOLD:
                                # Validates the negative distance is greater than the threshold
                                    totalNeg += 1
                                    correctNeg += 1
                                else:
                                    totalNeg += 1
                                # Calculates the negative accuracy 
                                    
                                print('Positive: ' + str(correctPos) + " / " + str(totalPos))
                                print('Negative: ' + str(correctNeg) + " / " + str(totalNeg))
                                print('Total: ' + str(correctPos + correctNeg) + " / " + str(totalPos + totalNeg))


class faceNet(tf.keras.Model):
# Defining the class that contains the faceNet architecture
    def __init__(self):
    # Procedure to initialise the inception layers
        super().__init__()

        self.vgg16 = tf.keras.applications.vgg16.VGG16(
            include_top=False, 
            weights='imagenet',
            input_shape=(224, 224, 3),
            pooling='avg'
        )

        self.faceNet = tf.keras.Sequential([
        # Creates a Keras Sequential that instantiates all the layers for the faceNet architecture 
            self.vgg16,
            tf.keras.layers.Dense(units=128, activation='relu'),
            tf.keras.layers.Lambda(lambda x: tf.math.l2_normalize(x, axis=1))
        ])

    def call(self, tensor):
    # Function to pass the anchor, positive, negative images through the model
        tensor = self.faceNet(tensor) 

        return tensor

annotationFilePath = 'identity_CelebA.txt'
# Initialises the path to the annotation file 

imagePathList, celebIDList = dataExtactor(annotationFilePath)   
imageList = tripletImage(imagePathList, celebIDList)
