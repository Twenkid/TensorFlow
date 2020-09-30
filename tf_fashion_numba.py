#@title MIT License
#
# Copyright (c) 2017 François Chollet
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

"""

 Training "fashion_mnist"

Tried and Modified by Todor Arnaudov, 8.4.2020,
 Added NUMBA, set CPU, release allocated memory
 python -m pip install numba
 For GPU/CUDA use download CUDA library and cuDNN 
 cuDNN installation is just an extraction of the library files
 If you get an error such as "Could not load dynamic library 'cudnn64_7.dll'; dlerror: cudnn64_7.dll not found:
 In my and other cases including the path in the %PATH% Environment var didn't help (Windows).
 The solution that worked was to copy the DLL to the /bin folder of the main CUDA installation, such as:
 C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v10.1\\bin
 
 Unexpectedly at first the CPU implementation was faster than on the GPU (even though a modest one - 750 Ti, but the CPU was not special as well, 4 cores/threads i5-6500). With a similar test on MNIST, the CPU was about twice as fast. Notice that the GPU was utilised just about 30-32% though.
 CPU ~ 25 s
 GPU ~ 45.8 s
""" 

# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras

# Helper libraries
import numpy as np
import matplotlib.pyplot as plt
from time import time

print(tf.__version__)


#### USE CPU if you wish etc. 

#python -m pip install numba
from numba import cuda 
device = cuda.get_current_device()
device.reset()
cuda.current_context().trashing.clear()

cpu = True #False
cpu = False

if cpu:
  #my_devices = #tf.config.experimental.list_physical_devices(device_type='CPU')
  #tf.config.experimental.set_visible_devices(devices= my_devices, device_type='CPU')
  #for anyone who is using tf 2.1, the above comment does not seems to work.
  print("Using CPU!")
  tf.config.set_visible_devices([], 'GPU')

tf.debugging.set_log_device_placement(True)


fashion_mnist = keras.datasets.fashion_mnist

(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
               
for n,cl in enumerate(class_names,start=0):
  print(n, cl)
 
			   
print(train_images.shape)
print(len(train_labels))
print(train_labels)
print(test_images.shape)
print(len(test_images.shape))

#PREPROCESS

plt.figure()
plt.imshow(train_images[0])
plt.colorbar()
plt.grid(True) #plt.grid(False)
plt.show()

#Normalize, BW 255 to 0.0 - 1.0 FLOAT
train_images = train_images / 255.0

test_images = test_images / 255.0

rowIm = 7
colIm = 7
numImages = rowIm * colIm
nEpochs = 10 #10
plt.figure(figsize=(10,10))
for i in range(rowIm*colIm):
    #plt.subplot(5,5,i+1)
    plt.subplot(rowIm,colIm,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_images[i], cmap=plt.cm.binary)
    plt.xlabel(class_names[train_labels[i]])
plt.show()

#BUILD MODEL

model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.3), #added - that solved the overfitting, correctly recognized Sneakers, 0.50 confidence (confused with Sandals - really similar, sandals have a hole 
    keras.layers.Dense(10)  #Number of classes
])

#COMPILE MODEL

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])
			  
        
start = time()
#TRAIN, Feed the model
model.fit(train_images, train_labels, epochs=nEpochs)
end = time()
print ("TIME: ", end-start)


#Evaluate accuracy
test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)
print('\nTest accuracy:', test_acc)

#Make predictions

#With the model trained, you can use it to make predictions about some images. The model's linear outputs, logits. Attach a #softmax layer to convert the logits to probabilities, which are easier to interpret.

probability_model = tf.keras.Sequential([model,tf.keras.layers.Softmax()])

predictions = probability_model.predict(test_images)

#Here, the model has predicted the label for each image in the testing set. Let's take a look at the first prediction:

print(predictions[0])

#A prediction is an array of 10 numbers. They represent the model's "confidence" that the image corresponds to each of the #10 different articles of clothing. You can see which label has the highest confidence value:

print(np.argmax(predictions[0]))

#So, the model is most confident that this image is an ankle boot, or class_names[9]. Examining the test label shows that #this classification is correct:

print(test_labels[0])

#Display



def plot_image(i, predictions_array, true_label, img):
  predictions_array, true_label, img = predictions_array, true_label[i], img[i]
  plt.grid(False)
  plt.xticks([])
  plt.yticks([])

  plt.imshow(img, cmap=plt.cm.binary)

  predicted_label = np.argmax(predictions_array)
  if predicted_label == true_label:
    color = 'blue'
  else:
    color = 'red'

  plt.xlabel("{} {:2.0f}% ({})".format(class_names[predicted_label],
                                100*np.max(predictions_array),
                                class_names[true_label]),
                                color=color)

def plot_value_array(i, predictions_array, true_label):
  predictions_array, true_label = predictions_array, true_label[i]
  plt.grid(False)
  plt.xticks(range(10))
  plt.yticks([])
  thisplot = plt.bar(range(10), predictions_array, color="#777777")
  plt.ylim([0, 1])
  predicted_label = np.argmax(predictions_array)

  thisplot[predicted_label].set_color('red')
  thisplot[true_label].set_color('blue')


#Verify predictions 
#With the model trained, you can use it to make predictions about some images.

#Let's look at the 0th image, predictions, and prediction array. Correct prediction labels are blue and incorrect prediction #labels are red. The number gives the percentage (out of 100) for the predicted label.

i = 0
plt.figure(figsize=(6,3))
plt.subplot(1,2,1)
plot_image(i, predictions[i], test_labels, test_images)
plt.subplot(1,2,2)
plot_value_array(i, predictions[i],  test_labels)
plt.show()

#In [0]:

i = 12
plt.figure(figsize=(6,3))
plt.subplot(1,2,1)
plot_image(i, predictions[i], test_labels, test_images)
plt.subplot(1,2,2)
plot_value_array(i, predictions[i],  test_labels)
plt.show()

#Let's plot several images with their predictions. Note that the model can be wrong even when very confident.


# Plot the first X test images, their predicted labels, and the true labels.
# Color correct predictions in blue and incorrect predictions in red.
num_rows = 5
num_cols = 3
num_images = num_rows*num_cols
plt.figure(figsize=(2*2*num_cols, 2*num_rows))
for i in range(num_images):
  plt.subplot(num_rows, 2*num_cols, 2*i+1)
  plot_image(i, predictions[i], test_labels, test_images)
  plt.subplot(num_rows, 2*num_cols, 2*i+2)
  plot_value_array(i, predictions[i], test_labels)
plt.tight_layout()
plt.show()

#Use the trained model

#Finally, use the trained model to make a prediction about a single image.

# Grab an image from the test dataset.
img = test_images[1]
print(img.shape)

#tf.keras models are optimized to make predictions on a batch, or collection, of examples at once. Accordingly, even though #you're using a single image, you need to add it to a list:

# Add the image to a batch where it's the only member.
img = (np.expand_dims(img,0))

print(img.shape)

#Now predict the correct label for this image:

predictions_single = probability_model.predict(img)

print(predictions_single)

#In [0]:

plot_value_array(1, predictions_single[0], test_labels)
_ = plt.xticks(range(10), class_names, rotation=45)

#keras.Model.predict returns a list of lists—one list for each image in the batch of data. Grab the predictions for our #(only) image in the batch:
#In [0]:

print(np.argmax(predictions_single[0]))

print ("TIME: ", end-start)

#And the model predicts a label as expected.
