from faceDetection import faceDetectionModel
import tensorflow as tf
import random
import os
 
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
    visited = []
    # Initialises a validation list to ensure we are not repeating a person
    imageList = []
    # List that contains all concatenated images to be put in the dataset
    counter = 0
    # Intialises counter variable that splits up dataset

    for celebID in celebIDList:
    # Iterates through each celebID
        if celebID not in visited:
        # Validation to check if we have already selected this individual
            
            counter += 1
            if counter > 0 and counter <= 2000:
            # Only processes a certain number of images at onces

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
                        
                                concatenatedImage = tf.concat([tf.reshape(anchor, [224, 224, 3]), tf.reshape(positive, [224, 224, 3]), tf.reshape(negative, [224, 224, 3])], axis=0)
                                # Concatenates the images so they count as 1 image
                                imageList.append(concatenatedImage)
    return imageList

if __name__ == '__main__':
                
    annotationFilePath = 'identity_CelebA.txt'
    # Initialises the path to the annotation file     

    imagePathList, celebIDList = dataExtactor(annotationFilePath)   
    imageList = tripletImage(imagePathList, celebIDList)
    annotationList = tf.zeros(len(imageList))

    tf.data.Dataset.save(tf.data.Dataset.from_tensor_slices((imageList, annotationList)), "dataset1")

