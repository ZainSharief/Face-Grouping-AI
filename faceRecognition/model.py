import tensorflow as tf

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
        anchor, positive, negative = tf.split(tensor, num_or_size_splits=3, axis=1)
        # Splits up the concatenated images into their respective images
        anchor, positive, negative = [self.faceNet(tensor) for tensor in [anchor, positive, negative]]
        # Individually passes each image through the model

        return tf.concat([anchor, positive, negative], axis=1)

def tripletLoss(y_true, y_pred):
# Function to calculate the triplet loss
    ALPHA = 0.3
    # Initialises the constant used in the triplet loss

    y_pred = tf.cast(y_pred, tf.float32)
    # Converts the model output to a float

    anchor, positive, negative = tf.split(y_pred, num_or_size_splits=3, axis=1)
    # Splits up the array of outputs into their respective output

    posDistance = tf.reduce_sum(tf.square(tf.subtract(anchor, positive)), axis=-1)
    negDistance = tf.reduce_sum(tf.square(tf.subtract(anchor, negative)), axis=-1)

    # Compute triplet loss
    loss = tf.maximum(0.0, posDistance - negDistance + ALPHA)

    return tf.reduce_mean(loss)

if __name__ == '__main__':

    trainDataset = tf.data.Dataset.load('trainDataset').batch(32)
    valDataset = tf.data.Dataset.load('valDataset').batch(32)

    model = faceNet()
    # Creates an instance of the model

    model.compile(
        loss=tripletLoss,
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
    )
    # Complies the model by declaring the loss function and optimizer

    cp_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath="faceRecognitionWeights1/cp.ckpt",
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

    model.fit(
        trainDataset,
        batch_size=32,
        epochs=30,
        validation_data=valDataset,
        callbacks=[cp_callback, early_stopping],
    )
    # Trains the model on the given dataset
