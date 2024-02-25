import tensorflow as tf
import os
import numpy as np
from PIL import Image

def preprocessImage(path):
# Function to preprocess the image
    image = Image.open(path)
    # Opens the image at the specified path
    image = image.resize((300, 300))
    # Resizing the image to 300x300 pixels
    image = np.array(image, dtype=np.float32)
    # Converting the image into a tensor 
    image = image / 255.0
    # Normalising the image [0-1]
    return image

def preprocessAnnotation(line, img):
# Function to preprocess the annotations
    bbox = line.strip().split(' ')
    # Initialises an array that contains each element in the line
    bbox = bbox[:4]
    # Removes everything except x1, y1, w, h
    bbox = [float(box) for box in bbox]
    # Converts bbox values into floats

    img = Image.open(img)
    # Opens the image using the Pillow library 
    imgWidth, imgHeight = img.size
    # Finds the width and height of the image
    bbox[0] = bbox[0] / imgWidth
    bbox[1] = bbox[1] / imgHeight
    bbox[2] = bbox[2] / imgWidth
    bbox[3] = bbox[3] / imgHeight
    # Normalises bbox values 0-1 

    return bbox

def bboxExtractor(annotationFilePath, ImageFilePath):
# Function to extract the training data values from the respective files
    imageList = []
    # Array to hold the preprocessed image tensors
    bboxList = []
    # Array to hold the preprocessed bounding box coordinates

    with open(annotationFilePath) as file:
    # Opens the annotation file
        iterator = iter(file)
        # Defines an iterator for a file
        for line in file:
        # Iterates through every line in the file
            line = line.strip()
            # Removes any unnecessary spaces at the end of the line
            if line.endswith('.jpg'):
            # Checks if the line ends with '.jpg' 
                imageBboxes = []
                # Initialises the list that will hold all the bounding boxes per image / clears it for each new image

                currentImage = os.path.join(ImageFilePath, line)
                # Combines the path to the image folder with the image path to make the full path
                try:
                    Image.open(currentImage).verify() 
                    imageList.append(preprocessImage(currentImage))
                    # Adds the preprocessed imagge to imageList list
                    
                    numPeople = int(next(iterator).strip())
                    # Looks at the next line to calculate how many people are in the current image
                    for i in range(numPeople):
                    # Iterates through each person
                        imageBboxes.append(preprocessAnnotation(next(iterator), currentImage))
                        # Preprocesses the annotations into the correct format
                    bboxList.append(imageBboxes)
                    # Appends a list of all the bboxes for an image to a list 
                
                except Exception as e:
                    print(line, 'returns the following error:', e)
    
    return imageList, bboxList

def anchorBoxGeneration(featureMaps, sizes, aspectRatios):
# Function to generate the anchor boxes
    anchorBoxes = []
    # Initalises a list to store all anchor boxes
    for i, feature in enumerate(featureMaps):
    # Iterates through each feature map
        for y in range(feature[0]):
        # Iterates across the y-axis for each feature map
            for x in range(feature[1]):
            # Iterates across the x-axis for each feature map
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

                        anchorBoxes.append([x1, y1, w, h])
                        # Appends the anchor box to the list of all anchor boxes 

    return anchorBoxes

def iouCalculator(box, anchor):
# Function to calculate the intersection over union between the two boxes
    x1A, y1A, wA, hA = box
    x1B, y1B, wB, hB = anchor
    # Exctracts the x1, y1, w, h from box and anchor list

    intersectionWidth = min(x1A + wA, x1B + wB) - max(x1A, x1B)
    intersectionHeight = min(y1A + hA, y1B + hB) - max(y1A, y1B)
    # Calculates the intersection width and height by finding the max and min of box coordinates

    if intersectionWidth <= 0 or intersectionHeight <= 0: return 0
    # Validates both are >= 0 and returns 0 if False
        	
    intersectionArea = intersectionWidth * intersectionHeight
    box1Area = wA * hA
    box2Area = wB * hB
    # Calculates the intersection area, box1 and box2 area

    unionArea = box1Area + box2Area - intersectionArea
    iou = intersectionArea / unionArea
    # Finds the intersection area and calculates IoU

    return iou

def assignAnchors(anchorBoxes, bboxList):
    THRESHOLD = 0.15
    # Initiates value that shows the minimum amount of intersection to consider positive

    classes = []
    # Initialises array that will hold all the class annotations
    localisation = []
    # Initialises array that will hold all the localisation (bounding box offset) annotations

    for bbox in bboxList:
    # Iterates through the set of bounding box coordinates per image
        classifcationLabels = np.zeros((len(anchorBoxes), 1))
        # Creates a 2d array of 0s with shape (numAnchors, numClasses) for each image
        localisationLabels = np.zeros((len(anchorBoxes), 4))
        # Creates a 2d array of 0s where the bounding box offsets will be

        for box in bbox:
        # Iterates through all the faces in an image.

            iou = [iouCalculator(box, anchor) for anchor in anchorBoxes]
            # Creates a list of IoU values with each anchor box 

            if max(iou) > THRESHOLD:
                classifcationLabels[np.argmax(iou)] = 1
                # Finds the position of the best anchor in the image and sets face class to positive
                localisationLabels[np.argmax(iou)] = offset(anchorBoxes[np.argmax(iou)], box)
                # Finds the position of the best anchor in the image and finds the bounding box offset

        classes.append(classifcationLabels)
        localisation.append(localisationLabels)
        # Appends the labels of that image to the annotation lists

    return tf.concat([localisation, classes], axis=2)

def offset(anchor, bbox):
# Function to calculate the bounding box offset
    x1A, y1A, wA, hA = bbox
    cx1A = x1A + (wA * 0.5)
    cy1A = y1A + (hA * 0.5)

    x1B, y1B, wB, hB = anchor
    cx1B = x1B + (wB * 0.5)
    cy1B = y1B + (hB * 0.5)
    # Calculates the centre coordinates of the anchor box and bbox 

    cx = (cx1A - cx1B) / wB
    cy = (cy1A - cy1B) / hB
    # Calculates the centre offset coordinates

    w = np.log(wA / wB)
    h = np.log(hA / hB)
    # Calculates the width and height offset coordinates

    return [cx, cy, w, h]

FEATUREMAPS = [(37, 37), (18, 18), (9, 9), (5, 5), (3, 3), (1, 1)]
ASPECTRATIOS = [[1.0, 0.8], [1.0, 0.8], [1.0, 0.8, 0.6], [1.0, 0.8, 0.6], [1.0, 0.8], [1.0, 0.8]]
SIZES = [[0.1, 0.2], [0.1, 0.2, 0.4, 0.6], [0.1, 0.2, 0.4, 0.6], [0.1, 0.2, 0.4, 0.6], [0.1, 0.2, 0.4, 0.6], [0.1, 0.2]]

if __name__ == '__main__':

    TRAINannotationFilePath = r'\WIDERFaces\wider_face_train_bbx_gt.txt'
    VALannotationFilePath = r'\WIDERFaces\wider_face_val_bbx_gt.txt'
    # Defines the name of the file that contains annotation data
    TRAINImageFilePath = r'.\WIDERFaces\WIDER_train\images'
    VALImageFilePath = r'.\WIDERFaces\WIDER_val\images'
    # Defines the folder path to access the dataset images

    anchorBoxes = anchorBoxGeneration(FEATUREMAPS, SIZES, ASPECTRATIOS)
    # Calls the anchor box generation function that generates anchor boxes given the above hyperparamters

    TRAINimageList, TRAINbboxList = bboxExtractor(TRAINannotationFilePath, TRAINImageFilePath)
    VALimageList, VALbboxList = bboxExtractor(VALannotationFilePath, VALImageFilePath)
    # Calls the function that extracts the bounding box information and preprocesses images

    TRAINannotations = assignAnchors(anchorBoxes, TRAINbboxList)
    VALannotations = assignAnchors(anchorBoxes, VALbboxList)
    # Calls the assign anchors function that assigns the positive anchor boxes
    
    VALdataset = tf.data.Dataset.from_tensor_slices((VALimageList, VALannotations))
    # Combine the images and labels to create the complete datasets
    tf.data.Dataset.save(VALdataset, 'valDataset')
    # Saves the dataset to a file

    TRAINdataset1 = tf.data.Dataset.from_tensor_slices((TRAINimageList[:4000], TRAINannotations[:4000]))
    TRAINdataset2 = tf.data.Dataset.from_tensor_slices((TRAINimageList[4000:8000], TRAINannotations[4000:8000]))
    TRAINdataset3 = tf.data.Dataset.from_tensor_slices((TRAINimageList[8000:], TRAINannotations[8000:]))
    # Combine the images and labels to create the complete datasets
    tf.data.Dataset.save(TRAINdataset1.concatenate(TRAINdataset2.concatenate(TRAINdataset3)), 'trainDataset')
    # Saves the dataset to a file

    print("Pre-processing Complete")

'''
Depending on the device used, the datasets may not save properly due to memory restrictions. 
To avoid this, split up the dataset into multiple smaller datasets and save these individually.
These can then be combined using the dataset1.concatenate(dataset2) function
'''
