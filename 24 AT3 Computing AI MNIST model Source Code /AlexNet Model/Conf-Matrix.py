import tensorflow as tf
from sklearn.metrics import confusion_matrix
import numpy as np
from tensorflow import keras
import tensorflow_datasets as tfds

# Data loading
(ds_train, ds_test), ds_info = tfds.load(
    'mnist',
    split=['train', 'test'],
    shuffle_files=True,
    as_supervised=True,
    with_info=True,
)

# Preprocess function to normalize the dataset
def preprocess(image, label):
    image = tf.cast(image, tf.float32) / 255.0  # Normalize to [0,1]
    label = tf.one_hot(label, depth=10)         # One-hot encode labels
    return image, label

# Apply preprocessing to the datasets and batch
ds_test = ds_test.map(preprocess).batch(32)

# Collect all images and labels from the test dataset
x_test = []
y_test = []

for image_batch, label_batch in ds_test:
    x_test.append(image_batch.numpy())
    y_test.append(label_batch.numpy())

# Ensure that the length of x_test and y_test are the same
x_test = np.concatenate(x_test)
y_test = np.concatenate(y_test)

# Convert the labels from one-hot encoded to class indices
y_test = np.argmax(y_test, axis=1)

# Loading and recompiling the model 
model = keras.models.load_model('AlexNet-A100.h5', compile=False)
model.compile(optimizer="adam", loss='categorical_crossentropy', metrics=['accuracy'])

# Predict on the test set
y_pred = np.argmax(model.predict(x_test), axis=-1)

# Calculate the confusion matrix
cm = confusion_matrix(y_test, y_pred)

# Extract True Positives (TP), False Positives (FP), False Negatives (FN), and True Negatives (TN)
TP = np.diag(cm)
FP = np.sum(cm, axis=0) - TP
FN = np.sum(cm, axis=1) - TP
TN = np.sum(cm) - (FP + FN + TP)

# Calculate rates
sensitivity = TP / (TP + FN)  # True Positive Rate (Sensitivity)
specificity = TN / (TN + FP)  # True Negative Rate (Specificity)
false_positive_rate = FN / (FN + TP)  # False Positive Rate (Type I Error)
false_negative_rate = FP / (FP + TN)  # False Negative Rate (Type II Error)

# Print results for each class
for i in range(10): 
    print(f"Class {i}:")
    print(f"True Positive: {TP[i]}")
    print(f"True Negative: {TN[i]}")
    print(f"False Positive: {FP[i]}")
    print(f"False: Negative: {FN[i]}")
    print(f"Sensitivity (True Pos): {sensitivity[i]*100:.2f}%")
    print(f"Specificity (True Neg): {specificity[i]*100:.2f}%")
    print(f"Type I Err (FP): {false_positive_rate[i]*100:.2f}%")
    print(f"Type II Err (FN): {false_negative_rate[i]*100:.2f}%")
    print("-------------------------------------------------------")
