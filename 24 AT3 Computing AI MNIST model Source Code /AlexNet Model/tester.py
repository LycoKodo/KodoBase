import tensorflow as tf
import numpy as np
from tensorflow import keras
import tensorflow_datasets as tfds

# data loading
(ds_train, ds_test), ds_info = tfds.load(
    'mnist',
    split=['train', 'test'],
    shuffle_files=True,
    as_supervised=True,
    with_info=True,
)

# Preprocessing data
    #data - float32
    # label - one hot encryption
def preprocess(image, label):
    image = tf.cast(image, tf.float32) / 255.0  # Normalize to [0, 1]
    label = tf.one_hot(label, depth=10)
    return image, label

ds_test = ds_test.map(preprocess).batch(32)


# Loading and recompiling model 
model = keras.models.load_model('AlexNet-FP64.h5', compile=False)
model.compile(optimizer=keras.optimizers.RMSprop(), loss='categorical_crossentropy', metrics=['accuracy'])

# Evaluating the model
loss, accuracy = model.evaluate(ds_test, verbose=2)

print("Loss: " + str(loss))
print("Accuracy: " + str(accuracy))