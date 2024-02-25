import tensorflow as tf

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
        featureMapsClass = [tf.reshape(fmap, (tf.shape(fmap)[0], -1, 1)) for fmap in [fmap37Class, fmap18Class, fmap9Class, fmap5Class, fmap3Class, fmap1Class]]
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
    
def SSDLoss(y_true, y_pred):
# Function that calculates the localisation and classification loss
    BETA = 0.02
    # Introduces a scale factor for localisation loss

    y_trueLoc = tf.cast(y_true[:, :, :4], tf.float32)
    y_trueClass = tf.cast(y_true[:, :, -1:], tf.float32)
    # Seperates the training data into localisation and classification

    y_predLoc = tf.cast(y_pred[:, :, :4], tf.float32)
    y_predClass = tf.cast(y_pred[:, :, -1:], tf.float32)
    # Seperates the model output into localisation and classification

    classLoss = tf.keras.losses.BinaryFocalCrossentropy(alpha=0.9)(y_trueClass, y_predClass)
    # Calculates the classification loss using Binary Focal Cross-Entropy
    locLoss = tf.keras.losses.Huber(delta=1)(y_trueLoc, y_predLoc)
    # Calculates the localisation loss using Huber loss

    loss = classLoss + (locLoss * BETA)
    # Calcultes the total loss

    return loss

if __name__ == '__main__':

    trainDataset = tf.data.Dataset.load('trainDataset').batch(32)
    valDataset = tf.data.Dataset.load('valDataset').batch(32)
    # Imports the training data from files

    model = SSD()
    # Creates an instance of the model

    model.compile(
        loss=SSDLoss,
        optimizer=tf.keras.optimizers.Adam(),
    )
    # Complies the model by declaring the loss function and optimizer

    cp_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath="faceDetectionWeights/cp.ckpt",
        save_best_only=True,
        save_weights_only=True,
    )
    # Checkpoint callback that saves model after every epoch if it performs better

    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=3,
        restore_best_weights=True,
    )
    # Callback that stops the model from training if it starts to overfit

    def scheduler(epoch, lr):
        if epoch < 10:
            return lr
        else:
            return lr * tf.math.exp(-0.1)
    # Function that schedules the learning rate through the epochs 

    lr_scheduler = tf.keras.callbacks.LearningRateScheduler(
        scheduler, 
        verbose=0
    )
    # Callback that reduces the learning rate over time

    model.fit(
        trainDataset,
        batch_size=32,
        epochs=30,
        validation_data=valDataset,
        callbacks=[cp_callback, early_stopping, lr_scheduler],
    )
    # Trains the model on the given dataset
