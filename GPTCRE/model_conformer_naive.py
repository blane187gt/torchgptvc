import tensorflow as tf
from tensorflow.keras import layers, models

class ConformerNaive(tf.keras.Model):
    def __init__(self):
        super(ConformerNaive, self).__init__()
        self.conv1 = layers.Conv2D(32, (3, 3), activation='relu')
        self.pool1 = layers.MaxPooling2D((2, 2))
        self.flatten = layers.Flatten()
        self.dense1 = layers.Dense(128, activation='relu')
        self.dense2 = layers.Dense(88, activation='softmax')  # Assuming 88 pitch classes

    def call(self, inputs):
        x = self.conv1(inputs)
        x = self.pool1(x)
        x = self.flatten(x)
        x = self.dense1(x)
        return self.dense2(x)
