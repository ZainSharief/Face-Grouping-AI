import pygame
import numpy as np
from tkinter import messagebox
from filesSQL import inputImages, process, deletePhoto, mergePeople, manuallyAdd, manuallyRemove, changeName, getFaces, getImages, getNonface, getPeopleInImage, getName

class button():
# Class for a button
    def __init__(self, x, y, width, height, text, colour, hovercolour, screen, textSize=50):
    # Function to set up the details about a button
        self.x = x 
        self.y = y
        self.width = width
        self.height = height
        self.colour = colour
        self.text = text
        self.textSize = textSize
        self.hovercolour = hovercolour
        self.screen = screen
        # Initialises the encapsulated variables

    def draw(self, scroll_y):
    # Function to check for any updates with the button
        
        self.border = pygame.draw.rect(self.screen, (0, 0, 0), pygame.Rect(self.x - 1, self.y - 1, self.width + 2, self.height + 2))
        self.button = pygame.draw.rect(self.screen, self.colour, pygame.Rect(self.x, self.y, self.width, self.height))
        # Draws the button onto the screen

        position = pygame.mouse.get_pos()
        position = (position[0], position[1] - scroll_y)
        # Finds the position of the user's mouse
        if self.button.collidepoint(position):
        # Checks if the mouse is hovering over the button 
            pygame.draw.rect(self.screen, self.hovercolour, pygame.Rect(self.x, self.y, self.width, self.height))
            # Draws the button onto the screen with the hover colour

        if self.text != '':
            font = pygame.font.SysFont(name='calibri', size=self.textSize, bold=True)
            # Initialises the font details used by the text
            text = font.render(self.text, 1, (0, 0, 0))
            # Renders the text as the colour black using antialias 
            self.screen.blit(text, (self.x + (self.width/2 - text.get_width()/2), self.y + (self.height/2 - text.get_height()/2)))
            # Blits the text onto the screen in the centre of the button

    def buttonClicked(self, scroll_y):
    # Function to see if the button has been clicked
        position = pygame.mouse.get_pos()
        position = (position[0], position[1] - scroll_y)
        # Finds the position of the mouse including during scrolling
        return self.button.collidepoint(position)
        # Returns True/False if the button has been clicked
    
class textBox():
# Class to set up a text box
    def __init__(self, x, y, width, height, text, colour, screen):
    # Function to set up the details about a button
        self.x = x 
        self.y = y
        self.width = width
        self.height = height
        self.colour = colour
        self.text = text
        self.screen = screen
        # Initialises the encapsulated variables

    def draw(self):
        self.border = pygame.draw.rect(self.screen, (0, 0, 0), pygame.Rect(self.x - 1, self.y - 1, self.width + 2, self.height + 2))
        self.textbox = pygame.draw.rect(self.screen, self.colour, pygame.Rect(self.x, self.y, self.width, self.height))
        # Draws the button onto the screen

        if self.text == '':
            font = pygame.font.SysFont(name='calibri', size=20)
            # Initialises the font details used by the text
            text = font.render('Search here...', 1, (0, 0, 0))
            # Renders the text as the colour black using antialias 
            self.screen.blit(text, (self.x + 10, self.y + (self.height/2 - text.get_height()/2)))
            # Blits the text onto the screen in the centre of the button

        else:
            font = pygame.font.SysFont(name='calibri', size=20)
            # Initialises the font details used by the text
            text = font.render(self.text, 1, (0, 0, 0))
            # Renders the text as the colour black using antialias 
            self.screen.blit(text, (self.x + 10, self.y + (self.height/2 - text.get_height()/2)))
            # Blits the text onto the screen in the centre of the button
    
    def buttonClicked(self, scroll_y):
    # Function to see if the button has been clicked
        position = pygame.mouse.get_pos()
        position = (position[0], position[1] - scroll_y)
        # Finds the position of the mouse
        return self.textbox.collidepoint(position)
        # Returns True/False if the button has been clicked

    def writeText(self, character):
    # Function to write text into the textBox
        if character == 8 and len(self.text) > 0:
        # Checks the character isn't backspace and validates the string isnt empty
            self.text = self.text[:-1]
            # Removes the last character
        elif len(self.text) <= 15 and (chr(character).isalpha() or character == 32):
        # Checks the string is not maximum character length
            self.text += chr(character)
            # Adds the character to the text
    
class searchBox(textBox):
# Class to set up the search box
    def __init__(self, x, y, width, height, text, colour, screen):
    # Procedure to initialise the search box
        super().__init__(x, y, width, height, text, colour, screen)
        # Calls the superclass to initialise the text box

class nameChange(textBox):
# Class to set up the search box
    def __init__(self, x, y, width, height, text, colour, screen):
    # Procedure to initialise the search box
        super().__init__(x, y, width, height, text, colour, screen)
        # Calls the superclass to initialise the text box

class inputButton(button):
# Creates the child class input button
    def __init__(self, x, y, width, height, text, colour, hovercolour, screen):
    # Procedure to initialise the button
        super().__init__(x, y, width, height, text, colour, hovercolour, screen)
        # Calls the init super class function to create a button instance

    def clicked(self, scroll_y):  
    # Procedure to update the button 
        if self.buttonClicked(scroll_y):
        # Checks if the button has been clicked
            inputImages()
            # Calls the input images function 

class goBack(button):
# Creates the child class input button
    def __init__(self, x, y, width, height, colour, hovercolour, screen, text='Go Back'):
    # Procedure to initialise the button
        super().__init__(x, y, width, height, text, colour, hovercolour, screen)
        # Calls the init super class function to create a button instance

    def clicked(self, currentState, scroll_y, newstate='main'):  
    # Procedure to update the button 
        if self.buttonClicked(scroll_y):
        # Checks if the button has been clicked
            return newstate
        else:
            return currentState
        
class MergeFaces(button):
# Creates the child class input button
    def __init__(self, x, y, width, height, colour, hovercolour, screen, text='Merge Faces'):
    # Procedure to initialise the button
        super().__init__(x, y, width, height, text, colour, hovercolour, screen, textSize=30)
        # Calls the init super class function to create a button instance

    def clicked(self, currentState, scroll_y):  
    # Procedure to update the button 
        if self.buttonClicked(scroll_y):
        # Checks if the button has been clicked
            return 'merge'
        else:
            return currentState
        
class AddPerson(button):
# Creates the child class add person button
    def __init__(self, x, y, width, height, screen, hovercolour=(200, 200, 200), colour=(255, 255, 255), text='People In Image  + '):
    # Procedure to initialise the button
        super().__init__(x, y, width, height, text, colour, hovercolour, screen, textSize=30)
        # Calls the init super class function to create a button instance
    
    def clicked(self, currentState, scroll_y):  
    # Procedure to update the button 
        if self.buttonClicked(scroll_y):
        # Checks if the button has been clicked
            return 'addPerson'
        else:
            return currentState
        
class RemovePerson(button):
# Creates the child class remove person button
    def __init__(self, x, y, width, height, screen, faceID, name, hovercolour=(200, 200, 200), colour=(255, 255, 255)):
    # Procedure to initialise the button
        self.faceID = faceID
        text = '       ' + name + ((18 - len(name))*' ') + 'x'
        # Makes the name string 18 characters long
        super().__init__(x, y, width, height, text, colour, hovercolour, screen, textSize=30)
        # Calls the init super class function to create a button instance
    
    def clicked(self, scroll_y):  
    # Procedure to update the button 
        if self.buttonClicked(scroll_y):
        # Checks if the button has been clicked
            return self.faceID
        
class deleteImage(button):
# Creates the child class input button
    def __init__(self, x, y, width, height, colour, hovercolour, screen, text='Delete Image'):
    # Procedure to initialise the button
        super().__init__(x, y, width, height, text, colour, hovercolour, screen, textSize=30)
        # Calls the init super class function to create a button instance

    def clicked(self, scroll_y):  
    # Procedure to update the button 
        if self.buttonClicked(scroll_y):
        # Checks if the button has been clicked
            return True
              
class imageButton(button):
# Creates the child class image button
    def __init__(self, x, y, width, height, screen, image, faceID, name):
        super().__init__(x, y, width, height, '', (255, 255, 255), (230, 230, 230), screen)
        # Calls the init super class function to create a button instance
        self.x = x
        self.y = y
        self.screen = screen
        self.image = image
        self.faceID = faceID
        self.name = name
        self.imgSize = image.get_size()

    def update(self, scroll_y):
    # Procedure to update the button 
        self.draw(scroll_y)
        # Draws on the button
        self.screen.blit(self.image, (self.x + ((self.width - self.imgSize[0]) / 2), self.y))
        # Places the image onto the screen in the centre of the button

        font = pygame.font.SysFont(name='calibri', size=30)
        # Initialises the font details used by the text
        name = font.render(self.name, 1, (0, 0, 0))
        # Renders the text as the colour black using antialias 
        self.screen.blit(name, (self.x + 0.5*(self.width - name.get_width()), self.y + 0.5*(self.imgSize[0] + self.height - name.get_height())))
        # Blits the text onto the screen in the centre of the button

    def clicked(self, scroll_y):
    # Function to see if a button has been clicked
        if self.buttonClicked(scroll_y):
        # Checks if the button has been clicked
            return self.faceID

class imageDisplay(button):
# Creates the child class image button
    def __init__(self, x, y, width, height, screen, image, imagePath):
        super().__init__(x, y, width, height, '', (255, 255, 255), (230, 230, 230), screen)
        # Calls the init super class function to create a button instance
        self.x = x
        self.y = y
        self.screen = screen
        self.image = image
        self.imagePath = imagePath
        self.imgSize = image.get_size()

    def update(self, scroll_y):
    # Procedure to update the button 
        self.draw(scroll_y)
        # Draws on the button
        self.screen.blit(self.image, (self.x + ((self.width - self.imgSize[0]) / 2), self.y))
        # Places the image onto the screen in the centre of the button

    def clicked(self, scroll_y):
    # Function to see if a button has been clicked
        if self.buttonClicked(scroll_y):
        # Checks if the button has been clicked
            return self.imagePath

def drawFaces(filter, xPosFace=190, yposFace=120):
# Function to generate the face buttons

    faceList = []
    # List to hold all the faces
    faceList.append(imageButton(x=20, y=yposFace, width=150, height=200, screen=screen, image=pygame.surfarray.make_surface(np.zeros((150, 150, 3))), faceID='nonperson', name='No People'))
    # Creates the button for no faces

    faceData = getFaces(filter)
    # Gets the exisiting faceID, name, thumbnails

    for faceID, name, thumbnail in faceData:
    # Iterates through the faceID, name, thumbnail for each face

        thumbnail = pygame.transform.rotate(pygame.surfarray.make_surface(thumbnail), 270)
        # Turns the image into a pygame surface and rotates it
        imgWidth, imgHeight = thumbnail.get_size()
        # Calculates the width and height of the image
        faceList.append(imageButton(x=xPosFace, y=yposFace, width=imgWidth, height=imgHeight+50, screen=screen, image=thumbnail, faceID=faceID, name=name))
        # Adds a new face to the list

        xPosFace += imgWidth + 20
        # Offsets the next image
        if xPosFace == 20 + (imgWidth + 20)*7:
        # Checks if 5 images have been placed on the screen
            xPosFace = 20
            yposFace += imgHeight + 70
            if yposFace > 7100:
                messagebox.showerror('Error', 'Too many photos, Remove some!')

    return faceList

def drawImages(clickedFace, xPosFace=20, yposFace=120):
# Function to generate the images for a face
    faceImageList = []
    # List to hold all the faces

    if clickedFace[0] == 'nonperson':
        imageList = [path for pathList in getNonface() for path in pathList]
        # Gathers all of the images of that person
    else:
        imageList = getImages(clickedFace)
        # Gathers all of the images of that person
        imageList = [imagepath for imagepathList in imageList for imagepath in imagepathList]

    if len(imageList) > 0:
        for imagePath in imageList:
            image = pygame.transform.scale(pygame.image.load(imagePath), (150,150))
            imgWidth, imgHeight = image.get_size()
            # Calculates the width and height of the image
            faceImageList.append(imageDisplay(x=xPosFace, y=yposFace, width=imgWidth, height=imgHeight, screen=screen, image=image, imagePath=imagePath))
            # Adds a new face to the list

            xPosFace += imgWidth + 20
            # Offsets the next image
            if xPosFace == 20 + (imgWidth + 20)*7:
            # Checks if 5 images have been placed on the screen
                xPosFace = 20
                yposFace += imgHeight + 20
                if yposFace > 7100:
                    messagebox.showerror('Error', 'Too many photos, Remove some!')

    return faceImageList

def drawSingleImage(clickedImage):
# Function to draw a singular image
    image = pygame.image.load(clickedImage[0])
    # Loads up the image to display
    imgWidth, imgHeight = image.get_size()
    # Calculates the width and height of the image

    if imgWidth > imgHeight:
    # Checks if the width is greater than the height
        imgHeight = imgHeight/(imgWidth/960)
        imgWidth = 960
        # Resizes the image while maintaining the aspect ratio
        image = pygame.transform.scale(image, (imgWidth, imgHeight))
        # Creates the image with the given size
    if imgWidth < imgHeight or imgHeight > 600:
        imgWidth = imgWidth/(imgHeight/600)
        imgHeight = 600
        # Resizes the image while maintaining the aspect ratio
        image = pygame.transform.scale(image, (imgWidth, imgHeight))
        # Creates the image with the given size

    screen.blit(image, (10 + (960 - imgWidth)/2, 110 + (600-imgHeight)/2))
    # Blits the image onto the screen

def drawPeopleInImage(clickedImage):
# Function to display all the people in an image
    faceDataList = getPeopleInImage(clickedImage)
    # Gets all the people in an image

    personButtonList = []
    imageThumbnailList = []
    yPos = 180
    # Initialises lists for the button and thumbnails

    for faceData in faceDataList:
        for faceID, name, thumbnail in faceData:
        # Iterates through the data to get the faceID, name, thumbnail
            personButtonList.append(RemovePerson(x=980, y=yPos, width=280, height=60, screen=screen, faceID=faceID, name=name))
            imageThumbnailList.append(pygame.transform.scale(pygame.transform.rotate(pygame.surfarray.make_surface(thumbnail), 270), (40,40)))
            # Creates an instance of the person button and the thumbnail
            yPos += 70
            # Offets the y position by 70

    return personButtonList, imageThumbnailList

def drawThumbnailsForImage(imageThumbnailList, screen):
# Function to draw the thumbnails next to the image buttons
    yPos = 180
    for image in imageThumbnailList:
    # Iterates through the thumbnails
        screen.blit(image, (990, yPos+10)) 
        # Blits the images onto the screen
        yPos += 70
        # Offsets the y position by 70 

pygame.init()
# Initialises pygame 

window = pygame.display.set_mode((1280, 720))
screen = pygame.surface.Surface((1280, 7200))
pygame.display.set_caption('Face Grouping Software')
clock = pygame.time.Clock()
scroll_y = 0
# Creates the screen and gives it a caption

inputBTN = inputButton(x=20, y=20, width=180, height=80, text='+ Input', colour=(230, 230, 230), hovercolour=(200, 200, 200), screen=screen)
# Creates an instance of a inputBTN
goBackBTN = goBack(x=20, y=20, width=260, height=80, colour=(230, 230, 230), hovercolour=(200, 200, 200), screen=screen)
# Creates an instance of goBack

deleteImageBTN = deleteImage(x=985, y=20, width=270, height=80, colour=(230, 230, 230), hovercolour=(200, 200, 200), screen=screen)
# Creates the button to delete an image from the software

mergeFacesBTN = MergeFaces(x=1080, y=20, width=180, height=80, colour=(230, 230, 230), hovercolour=(200, 200, 200), screen=screen)
person1 = None
# Creates an instance of MergeFaces

addPersonBTN = AddPerson(x=985, y=110, width=270, height=60, screen=screen)
# Creates an instance of the AddPerson class

searchBoxTBX = searchBox(x=240, y=20, width=720, height=80, text='', colour=(230, 230, 230), screen=screen)
searchBoxClicked = False
# Creates an instance of the search box

nameChangeTBX = nameChange(x=300, y=20, width=150, height=80, text='', colour=(230, 230, 230), screen=screen)
nameChangeClicked = False
# Creates an instance of the search box

state = 'main'
eventLoop = True    
while eventLoop:
# Initialises Event loop
    
    if state == 'main':
        faceList = drawFaces(searchBoxTBX.text)
        # Create all of the faces that will be shown on the screen

        screen.fill((245, 245, 245))
        # Makes the screen white

        inputBTN.draw(scroll_y)
        searchBoxTBX.draw()
        mergeFacesBTN.draw(scroll_y)
        [face.update(scroll_y) for face in faceList]
        # Draws all of the buttons on the screen and updates

        process()
        # Process all of the unsorted images

    elif state == 'face':
        imageList = drawImages(clickedFace)
        # Create all of the images that will be shown on the scree

        screen.fill((245, 245, 245))
        # Makes the screen white

        goBackBTN.draw(scroll_y)
        if clickedFace[0] != 'nonperson':
            nameChangeTBX.text = getName(clickedFace)
            nameChangeTBX.draw()
        # Draws on the current name into the text box
        [image.update(scroll_y) for image in imageList]

    elif state == 'image':
    # Checks if the state is image
        personInImageList, imageThumbnailList = drawPeopleInImage(clickedImage)
        # Generates the buttons and thumbnails

        screen.fill((245, 245, 245))
        # Makes the screen white

        goBackBTN.draw(scroll_y)
        deleteImageBTN.draw(scroll_y)
        # Draws on the main menu button and delete image button
        drawSingleImage(clickedImage)
        # Draws on the central image
        addPersonBTN.draw(scroll_y)
        # Draws on the add person button
        [personInImage.draw(scroll_y) for personInImage in personInImageList]
        drawThumbnailsForImage(imageThumbnailList, screen) 
        # Draws on the buttons and thumbnails
        
    elif state == 'merge':
        faceList = drawFaces(searchBoxTBX.text)
        # Create all of the faces that will be shown on the screen

        screen.fill((245, 245, 245))
        # Makes the screen white

        goBackBTN.draw(scroll_y)
        # Draws on the main menu button
        [face.update(scroll_y) for face in faceList]
        # Draws all of the buttons on the screen and updates

    elif state == 'addPerson':
        faceList = drawFaces(searchBoxTBX.text)
        # Create all of the faces that will be shown on the screen

        screen.fill((245, 245, 245))
        # Makes the screen white

        goBackBTN.draw(scroll_y)
        # Draws on the main menu button
        [face.update(scroll_y) for face in faceList]
        # Draws all of the buttons on the screen and updates

    for event in pygame.event.get():
    # Iterates through each pygame event
        
        if event.type == pygame.QUIT:
            eventLoop = False
        # Closes the software if the user clicks the x
    
        elif event.type == pygame.MOUSEBUTTONDOWN:
        # Detects the event of a click
                
            if event.button == 1:
                if state == 'main':
                # Checks if the state is main
                    clickedFace = [face.clicked(scroll_y) for face in faceList if face.clicked(scroll_y) != None]
                    # Finds the currently selected face

                    if len(clickedFace) != 0: currentFace = clickedFace 
                    # Updates a variable that holds the previously clicked face

                    if len(clickedFace):
                        state = 'face'
                        scroll_y = 0
                    # Checks if a face is clicked
                    inputBTN.clicked(scroll_y)
                    state = mergeFacesBTN.clicked(state, scroll_y)
                    # Calls the buttons when there is a click detected
                    
                    searchBoxClicked = searchBoxTBX.buttonClicked(scroll_y)
                    # Checks if the search box is clicked
                
                elif state == 'face':
                # Checks if the state is face
                    
                    clickedImage = [image.clicked(scroll_y) for image in imageList if image.clicked(scroll_y) != None]
                    # Finds the currently selected face   
                    state = goBackBTN.clicked(currentState='face', scroll_y=scroll_y)
                    # Updates the state when the main menu is clicked
                    if len(clickedImage) and state == 'face':
                        state = 'image'
                        scroll_y = 0
                    # Checks if a face is clicked
                        
                    if clickedFace[0] != 'nonperson':
                        nameChangeClicked = nameChangeTBX.buttonClicked(scroll_y)
                        # Checks if the search box is clicked
    
                elif state == 'image':
                # Checks if the state is image

                    clickedThumbnail = [image.clicked(scroll_y) for image in personInImageList if image.clicked(scroll_y) != None]
                    # Checked if any of the people have been clicked
                    
                    state = addPersonBTN.clicked(currentState=state, scroll_y=scroll_y)
                    # Updates the state when the button is clicked
                    state = goBackBTN.clicked(currentState=state, scroll_y=scroll_y, newstate='face')
                    # Updates the state when the main menu is clicked
                    
                    if len(clickedThumbnail):
                        manuallyRemove(clickedImage[0], clickedThumbnail[0])
                        # Remove them from the image

                    if deleteImageBTN.clicked(scroll_y):
                        state = 'main'
                        scroll_y = 0
                        deletePhoto(clickedImage[0])
                        # Deletes the current image if button is clicked

                    if state == 'face': clickedFace = currentFace
                    # Updates a variable that shows the next face as the one previously clicked on

                elif state == 'addPerson':
                # Checks if the state is adding a person
                    clickedFace = [face.clicked(scroll_y) for face in faceList if face.clicked(scroll_y) != None]
                    # Detects if any of the users are clicked
                    faceDataList = getPeopleInImage(clickedImage)
                    # Gets all of the people in the image
                    if len(clickedFace):
                        personExistsInImage = [True for facecData in faceDataList for faceID, name, thumbnail in facecData if clickedFace[0] == faceID]
                        if True in personExistsInImage:
                        # Checks if the selected person is already in the image
                            messagebox.showerror('Error', 'User is already in image')
                            # Prints an error if user selects individual already in the image
                            state = 'image'
                            scroll_y = 0
                            # Resets the state back to image
                        else:
                            manuallyAdd(clickedImage[0], clickedFace[0])
                            # Adds face to image
                            state = 'image'
                            scroll_y = 0
                            # Resets the state back to image

                    state = goBackBTN.clicked(currentState=state, scroll_y=scroll_y, newstate='image')
                    # Updates the state when the main menu is clicked
                    
                elif state == 'merge':
                # Checks if the state is 
                    clickedFace = [face.clicked(scroll_y) for face in faceList if face.clicked(scroll_y) != None]
                    # Finds the currently selected face
                    if person1 is None and len(clickedFace):
                    # Checks if there is already a person1
                        person1 = clickedFace[0]
                    elif len(clickedFace):
                        if person1 == clickedFace[0]:
                        # Checks if the user is trying to merge the same person
                            messagebox.showerror('Error', 'Cannot Merge the same person!')
                            # Prints an error if so
                        else:
                            mergePeople(person1, clickedFace[0])
                            # Calls the merge faces function
                        person1 = None
                        state = 'main'
                        scroll_y = 0
                    elif goBackBTN.clicked(currentState='merge', scroll_y=scroll_y) == 'main':
                    # Checks if the user wants to go back to main menu
                        person1 = None
                        state = 'main'
                        scroll_y = 0

            elif event.button == 4: 
                scroll_y = min(scroll_y + 20, 0)
            elif event.button == 5: 
                scroll_y = max(scroll_y - 20, -7200)
            # Sets up the scrolling function of the code

        elif event.type == pygame.KEYDOWN:
        # Detects the event of a character being clicked
            if state == 'main':
            # Checks if the state is main    
                if (event.key >= 65 and event.key <= 90) or (event.key >= 97 and event.key <= 122) or event.key == 8 or event.key == 32:   
                # Validates the user clicks an alpha character or backspace
                    if searchBoxClicked:
                    # Ensures the user is currently selecting the search bar
                        searchBoxTBX.writeText(event.key)
                        # Writes the character on the search bar

            if state == 'face':
            # Checks if the state is face    
                if (event.key >= 65 and event.key <= 90) or (event.key >= 97 and event.key <= 122) or event.key == 8 or event.key == 32:   
                # Validates the user clicks an alpha character or backspace
                    if nameChangeClicked:
                    # Ensures the user is currently selecting the search bar
                        nameChangeTBX.writeText(event.key)
                        # Writes the character on the search bar
                        changeName(clickedFace[0], nameChangeTBX.text)
                        # Changes the name of the individial
    
    window.blit(screen, (0, scroll_y), (0, 0, 1280, 7200))
    # Blits the screen onto the view window
    pygame.display.flip()
    # Updates the display
    clock.tick(60)

pygame.quit()
