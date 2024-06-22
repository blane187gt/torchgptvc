import tensorflow as tf
from tensorflow.keras import layers, models

class ConvNeXt(tf.keras.Model):
    def __init__(self):
        super(ConvNeXt, self).__init__()
        self.conv1 = layers.Conv2D(64, (3, 3), activation='relu')
        self.pool1 = layers.MaxPooling2D((2, 2))
        self.conv2 = layers.Conv2D(128, (3, 3), activation='relu')
        self.pool2 = layers.MaxPooling2D((2, 2))
        self.flatten = layers.Flatten()
        self.dense1 = layers.Dense(256, activation='relu')
        self.dense2 = layers.Dense(88, activation='softmax')  # Assuming 88 pitch classes

    def call(self, inputs):
        x = self.conv1(inputs)
        x = self.pool1(x)
        x = self.conv2(inputs)
        x = self.pool2(x)
        x = self.flatten(x)
        x = self.dense1(x)
        return self.dense2(x)
