import tensorflow as tf
import numpy as np
import tkinter as tk
from PIL import Image, ImageTk, ImageOps
import cv2
from io import BytesIO

from tensorflow import keras
from keras.utils import img_to_array
from keras.optimizers import RMSprop

# Load the AlexNet Model
model = keras.models.load_model('AlexNet-A100.h5', compile=False)
model.compile(optimizer="adam", loss='categorical_crossentropy')

class DrawingApp:
    def __init__(self, root):
        self.root = root
        self.root.title("AlexNet Handwritten Digit Classifier")
        
        self.canvas = tk.Canvas(self.root, width=280, height=280, bg='white')
        self.canvas.pack()

        self.result_label = tk.Label(self.root, text="[Predicted class will appear here]")
        self.result_label.pack()

        self.certainty = tk.Label(self.root, text="[Model's confidence will appear here]")
        self.certainty.pack()

        self.clear_button = tk.Button(self.root, text="Clear", command=self.clear_canvas)
        self.clear_button.pack()

        self.predict_button = tk.Button(self.root, text="Predict", command=self.predict_drawing)
        self.predict_button.pack()

        # Bind the mouse events to the canvas
        self.canvas.bind("<B1-Motion>", self.draw)

        # Store the drawing coordinates
        self.drawing_coords = []

    def draw(self, event):
        """ Captures the user's drawing stroke by stroke. """
        x, y = event.x, event.y
        radius = 8
        self.canvas.create_oval(x-radius, y-radius, x+radius, y+radius, fill="black")
        self.drawing_coords.append((x, y))

    def clear_canvas(self):
        """ Clears the canvas and resets the drawing. """
        self.canvas.delete("all")
        self.drawing_coords = []

    def predict_drawing(self):
        """ Converts the drawing into a 28x28 grayscale image and predicts the digit. """
        # Convert the canvas drawing to an image
        canvas_postscript = self.canvas.postscript(colormode='mono')
        img = Image.open(BytesIO(canvas_postscript.encode('utf-8')))
        
        # Convert to grayscale and invert colors (background black, digit white)
        img = img.convert('L')
        img = ImageOps.invert(img)

        # Resize to 28x28 pixels
        img = img.resize((28, 28))

        # Convert the image to an array and normalize it
        img_array = img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

        # Predict using the model
        prediction = model.predict(img_array)
        predicted_class = np.argmax(prediction)

        # Display the class probabilities
        class_probabilities = prediction[0] * 100
        class_probabilities_text = "\n".join([f"Class {i}: {prob:.2f}%" for i, prob in enumerate(class_probabilities)])

        # Update UI with prediction results
        self.certainty.config(text=f"Class probabilities (2 d.p.):\n{class_probabilities_text}")
        self.result_label.config(text=f"Predicted class: {predicted_class}")

# Initialize the Tkinter window
root = tk.Tk()
app = DrawingApp(root)
root.mainloop()
