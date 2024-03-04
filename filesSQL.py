from PIL import Image
from tkinter import filedialog, messagebox
import numpy as np
import tensorflow as tf
import shutil
import uuid
import os
import sqlite3
import cv2  

from model import faceGroupingModel

def __init__():
# Function to initialise the databases and folders
    if not os.path.exists(r'.\images'):
        os.makedirs(r'.\images')
    # Creates the images folder if it does not exist

    if not os.path.exists(r'.\images\unsorted'):
        os.makedirs(r'.\images\unsorted')
    # Creates the folder for unsorted images if it does not exist

    if not os.path.exists(r'.\images\sorted'):
        os.makedirs(r'.\images\sorted')
    # Creates the folder for sorted images if it does not exist
        
    conn = sqlite3.connect('faceGrouping.db')
    cursor = conn.cursor()
    # Opens and creates a connection to the database

    cursor.execute('''
    CREATE TABLE IF NOT EXISTS tblImages(
        ImagePath text,
        containsPerson BOOL
    )
    ''')
    # Creates the tblImages table if it does not exist
    conn.commit()

    cursor.execute('''
    CREATE TABLE IF NOT EXISTS tblFaces(
        faceID text,
        embedding BLOB,
        name text,
        thumbnail BLOB
    )
    ''')
    # Creates the tblImages table if it does not exist
    conn.commit()

    cursor.execute('''
    CREATE TABLE IF NOT EXISTS tblFacesInImage(
        ImagePath text,
        faceID text
    )
    ''')
    # Creates the tblImages table if it does not exist
    conn.commit()
    conn.close()

def inputImages():
# Procedure to input images

    invalid = []
    # Initalises list of invalid images

    imageFilePaths = filedialog.askopenfilenames(
        title="Input Images", 
        filetypes=[("Images", "*.jpg; *.png;")]
    )
    # Opens file explorer which allows us to select images

    for image in imageFilePaths:
    # Iterates through each image
        try:
            Image.open(image).verify() 
            # Verifies the image is valid

            newPath = os.path.join(r'.\images\unsorted', str(uuid.uuid4()) + os.path.splitext(image)[-1])
            # Combines the path to the unsorted folder with the a unique uuid to create a primary key
            shutil.copy(image, newPath)
            # Copies the image into the new path

            conn = sqlite3.connect('faceGrouping.db')
            cursor = conn.cursor()
            # Opens and creates a connection to the database
            cursor.execute('''
                INSERT INTO tblImages VALUES(
                    ?, ?
                )
            ''', (newPath, False))
            # Adds the details of the image to the database
            conn.commit()
            conn.close()

        except Exception as e:
            invalid.append(image)
            # Adds the image path to the invalid list if the image is invalid

    if len(invalid) > 0:    
        messagebox.showerror("Invalid Images", 'The following images could not be opened: ' +  ", ".join(invalid))
    # Displays an on-screen text message displaying the images are invalid
        
def process():
# Procedure to process each image
    
    THRESHOLD = 0.35
    unsortedImagePaths = os.listdir(r'.\images\unsorted')
    # Collects all unsorted images

    conn = sqlite3.connect('faceGrouping.db')
    cursor = conn.cursor()
    # Opens and creates a connection to the database

    for imagePath in unsortedImagePaths:
        OldimagePath = os.path.join(r'.\images\unsorted', imagePath)
        # Creates the full image path
        newImagePath = os.path.join(r'.\images\sorted', imagePath)
        # Creates the full new image path

        cursor.execute('''
            SELECT embedding FROM tblFaces
        ''')
        existingEmbeddings = cursor.fetchall()
        # Collects all exisiting embeddings for faces

        embeddingList, thumbnailList = faceGroupingModel(OldimagePath)
        # Pass our images through the model 

        if len(embeddingList) > 0:
        # Validates the image has at least 1 person in it 

            updateImage(newImagePath, OldimagePath, containsPerson=True)
            # Updates the path of the image and whether or not it contains a person

            for i, imageEmbedding in enumerate(embeddingList):
            # Iterates through each face in the image
    
                distanceList = []
                # Initialises a list that will hold the distances for each person

                for embeddingSet in existingEmbeddings:
                # Iterates through each person's embeddings
                    embeddingSet = [np.frombuffer(embedding, dtype=np.float32) for embedding in embeddingSet]
                    embeddingTensor = np.reshape(embeddingSet, (-1, 128))
                    embeddingSetList = np.split(embeddingTensor, embeddingTensor.shape[0], axis=0)
                    # Converts the value back into a tensor and splits it up by individual

                    distanceList.append(np.min([tf.reduce_sum(tf.square(tf.subtract(embeddding, imageEmbedding)), axis=-1) for embeddding in embeddingSetList]))
                    # Calculates the smallest distance of all the embeddings of a person 
                    
                if len(distanceList) != 0: 

                    if np.min(distanceList) <= THRESHOLD:
                    # Validates that the smallest value from distanceList is less than the threshold
                        addImageToPerson(newImagePath, existingEmbeddings[np.argmin(distanceList)])
                        # Updates table to add the individual to the image 
                        del existingEmbeddings[np.argmin(distanceList)]

                    else:
                        addPerson(imageEmbedding, newImagePath, thumbnailList[i])
                        # Adds a new person to the table
                else:   
                    addPerson(imageEmbedding, newImagePath, thumbnailList[i])
                    # Adds a new person to the table
        else:
            updateImage(newImagePath, OldimagePath, containsPerson=False)
            # Updates the path of the image and whether or not it contains a person
        
        shutil.move(OldimagePath, r'.\images\sorted')
        # Moves the image into the sorted folder

    conn.close()

def updateImage(newImagePath, OldimagePath, containsPerson=False):
# Updates the image path and whether or not it contains a person
    conn = sqlite3.connect('faceGrouping.db')
    cursor = conn.cursor()
    # Opens and creates a connection to the database

    cursor.execute('''
        UPDATE tblImages SET imagePath = ?, containsPerson = ? WHERE ImagePath = ?
    ''', (newImagePath, containsPerson, OldimagePath))
    # Updates the image with its new path and that it contains a person
    conn.commit()
    conn.close()

def addImageToPerson(newImagePath, imageEmbedding):
# Adds a person to an image
    conn = sqlite3.connect('faceGrouping.db')
    cursor = conn.cursor()
    # Opens and creates a connection to the database

    imageEmbedding = np.reshape([np.frombuffer(embedding, dtype=np.float32) for embedding in imageEmbedding], (-1, 128)).tobytes()

    faceID = cursor.execute('''
        SELECT faceID FROM tblFaces WHERE embedding = ?
    ''', (imageEmbedding,))
    # Finds the faceID of the value less than the threshold
    faceID = cursor.fetchall()[0][0]
    # Extracts the selected faceID

    cursor.execute('''
        INSERT INTO tblFacesInImage VALUES(
            ?, ?
        )
    ''', (newImagePath, faceID))
    # Inserts a link between a person and an image
    conn.commit()
    conn.close()

def addPerson(imageEmbedding, newImagePath, thumbnail):
# Adds a person to the database
    conn = sqlite3.connect('faceGrouping.db')
    cursor = conn.cursor()
    # Opens and creates a connection to the database

    faceID = str(uuid.uuid4())
    # Creates a unqiue identifier for that person

    imageEmbedding = np.reshape([np.frombuffer(embedding, dtype=np.float32) for embedding in imageEmbedding], (-1, 128)).tobytes()

    cursor.execute('''
        INSERT INTO tblFaces VALUES(
            ?, ?, ?, ?
        )
    ''', (faceID, imageEmbedding, 'unnamed', np.array(thumbnail, dtype=np.float32).tobytes()))
    # Inserts the new face into the database with the embedding
    conn.commit()
     
    cursor.execute('''
        INSERT INTO tblFacesInImage VALUES(
            ?, ?
        )
    ''', (newImagePath, faceID))
    # Inserts a link between a person and an image
    conn.commit()
    conn.close()

def deletePhoto(ImagePath):
# Deletes a photo from the system
    conn = sqlite3.connect('faceGrouping.db')
    cursor = conn.cursor()
    # Opens and creates a connection to the database

    cursor.execute('''
        SELECT faceID FROM tblFacesInImage WHERE ImagePath = ?
    ''', (ImagePath,))
    faceList = cursor.fetchall()
    faceList = [faceID for faceIDList in faceList for faceID in faceIDList]
    # Finds all faces in that image

    cursor.execute('''       
        DELETE FROM tblFacesInImage WHERE ImagePath = ?      
    ''', (ImagePath,))
    # Deletes any links with the image
    conn.commit()

    cursor.execute('''       
        DELETE FROM tblImages WHERE ImagePath = ?      
    ''', (ImagePath,))
    # Deletes the image from the table
    conn.commit()

    for faceID in faceList: 
    # Iterates through the faces
        cursor.execute('''
            SELECT ImagePath FROM tblFacesInImage WHERE faceID = ?
        ''', (faceID,))
        faceIDList = cursor.fetchall()
        # Extracts any images for that face

        if not len(faceIDList):
            cursor.execute('''       
                DELETE FROM tblFaces WHERE faceID = ?      
            ''', (faceID,))
            # Deletes the face from the table if there are no corresponding images
            conn.commit()

    conn.close()

    os.remove(ImagePath)
    # Removes the image file
    
def mergePeople(faceID1, faceID2):
# Merges together two people
    conn = sqlite3.connect('faceGrouping.db')
    cursor = conn.cursor()
    # Opens and creates a connection to the database

    cursor.execute('''
        UPDATE tblFacesInImage SET faceID = ? WHERE faceID = ? 
    ''', (faceID1, faceID2))
    conn.commit()
    # Updates the images that reference person2 to reference person1

    cursor.execute('''
        SELECT embedding FROM tblFaces WHERE faceID = ? OR faceID = ?
    ''', (faceID1, faceID2))
    imageEmbedding = cursor.fetchall()
    imageEmbedding = [embedding for embeddingSet in imageEmbedding for embedding in embeddingSet]
    # Fetches the embeddings and merges them into a 1D list

    cursor.execute('''
        UPDATE tblFaces SET embedding = ? WHERE faceID = ?
    ''', (np.array(imageEmbedding).tobytes(order='C'), faceID1))
    conn.commit()
    # Updates the table to hold the new embeddings

    cursor.execute('''       
        DELETE FROM tblFaces WHERE faceID = ?
    ''', (faceID2,))
    conn.commit()
    conn.close()
    # Deletes faceID2 from the table as it is no longer required
    
def manuallyAdd(ImagePath, faceID):
# Manually adds a person to an image
    conn = sqlite3.connect('faceGrouping.db')
    cursor = conn.cursor()
    # Opens and creates a connection to the database

    cursor.execute('''
        INSERT INTO tblFacesInImage VALUES(
            ?, ?
        )
    ''', (ImagePath, faceID))
    # Adds a connection between the person and the image

    cursor.execute('''
        UPDATE tblImages SET containsPerson = ? WHERE ImagePath = ?
    ''', (True, ImagePath))
    # Enforces that the image has a person in it

    conn.commit()
    conn.close()
    # Inserts a link between the face and image

def manuallyRemove(ImagePath, faceID):
# Manually removes a person from an image
    conn = sqlite3.connect('faceGrouping.db')
    cursor = conn.cursor()
    # Opens and creates a connection to the database

    cursor.execute('''       
        DELETE FROM tblFacesInImage WHERE ImagePath = ? AND faceID = ?
    ''', (ImagePath, faceID))
    conn.commit()
    # Deletes the connection between the user and image

    cursor.execute('''
        SELECT ImagePath FROM tblFacesInImage WHERE faceID = ?
    ''', (faceID,))
    imageList = cursor.fetchall()
    # Finds all the images that have that face

    if not imageList:
        cursor.execute('''       
            DELETE FROM tblFaces WHERE faceID = ?
        ''', (faceID,))
        conn.commit()
        # Deletes the face if there are no images with that face

    cursor.execute('''
        SELECT faceID FROM tblFacesInImage WHERE ImagePath = ?
    ''', (ImagePath,))
    faceList = cursor.fetchall()
    # Finds all faces in that image
   
    if not len(faceList):
        cursor.execute('''       
            UPDATE tblImages SET containsPerson = ? WHERE ImagePath = ?
        ''', (False, ImagePath))
        conn.commit()
    # Sets the image to nonface if it does not have any faces in it
        
    conn.close()
    # Deletes a link between the face and image

def changeName(faceID, name):
# Function to change the name of a face
    
    conn = sqlite3.connect('faceGrouping.db')
    cursor = conn.cursor()
    # Opens and creates a connection to the database

    if len(name) <= 15:
    # Validates the name is less than 15 characters
        cursor.execute('''
            UPDATE tblFaces SET name = ? WHERE faceID = ? 
        ''', (name, faceID))
        conn.commit()
        conn.close()
        # Updates the faces with both faceIDs to the new faceID 
    else:
        messagebox.showerror('Name is too long! Maximum 15 Characters')
        # Reports error if the name is too long

def getFaces(filter):
# Function to get the faceID, name and thumbnail of all faces
    
    conn = sqlite3.connect('faceGrouping.db')
    cursor = conn.cursor()
    # Opens and creates a connection to the database
    if filter != '':
        filter = '%' + filter + '%'
        cursor.execute('''
            SELECT faceID, name, thumbnail FROM tblFaces WHERE name LIKE ? AND name != ?
        ''', (filter, 'unnamed'))
    else:
        cursor.execute('''
            SELECT faceID, name, thumbnail FROM tblFaces
        ''')
    faceData = cursor.fetchall()
    conn.close()
    # Fetches the faceID, name and thumbnail of all faces  

    faceData = [[faceID, name, cv2.resize(np.frombuffer(thumbnail, dtype=np.float32).reshape(224, 224, 3), dsize=(150,150))] for faceID, name, thumbnail in faceData]
    # Converts the BLOB imagea into numpy as a 224x224x3 image

    return faceData

def getImages(faceID):
# Function to get the image paths
    
    conn = sqlite3.connect('faceGrouping.db')
    cursor = conn.cursor()
    # Opens and creates a connection to the database

    cursor.execute('''
        SELECT ImagePath FROM tblFacesInImage WHERE faceID = ? 
    ''', (faceID))
    imageList = cursor.fetchall()
    conn.commit()
    conn.close()
    # Gathers all of the image paths with that faceID

    return imageList

def getNonface():
# Function to get the image paths of images without a face
    conn = sqlite3.connect('faceGrouping.db')
    cursor = conn.cursor()
    # Opens and creates a connection to the database

    cursor.execute('''
        SELECT ImagePath FROM tblImages WHERE containsPerson = False
    ''')
    imageList = cursor.fetchall()
    conn.commit()
    conn.close()
    # Gathers all of the image paths with that faceID

    return imageList

def getPeopleInImage(ImagePath):
# Function to get the people in an image
    conn = sqlite3.connect('faceGrouping.db')
    cursor = conn.cursor()
    # Opens and creates a connection to the database

    cursor.execute('''
        SELECT faceID FROM tblFacesInImage WHERE ImagePath = ?
    ''', (ImagePath[0],))
    faceIDList = cursor.fetchall()
    faceIDList = [id for faceID in faceIDList for id in faceID]
    # Finds all the people in that image 

    faceDataList = []
    # Lists that contains the data for each face

    for faceID in faceIDList:
    # Iterates through each face in the image
        cursor.execute('''
            SELECT faceID, name, thumbnail FROM tblFaces WHERE faceID = ?
        ''', (faceID,))
        faceData = cursor.fetchall()
        faceDataList.append([[faceID, name, np.frombuffer(thumbnail, dtype=np.float32).reshape(224, 224, 3)] for faceID, name, thumbnail in faceData])
        # Extracts the faceID, name, thumbnail for that face and processes the image
    
    conn.close()

    return faceDataList
    
def getName(faceID):
# Function to get the name of a specific faceID
    conn = sqlite3.connect('faceGrouping.db')
    cursor = conn.cursor()
    # Opens and creates a connection to the database

    cursor.execute('''
        SELECT name FROM tblFaces WHERE faceID = ?
    ''', (faceID[0],))
    name = cursor.fetchall()[0][0]
    conn.close()
    # Fetches the name at the given faceID

    return name

__init__()

