import tensorflow as tf
#import numba
'''
>>> dir(tf.keras.datasets)
['__builtins__', '__cached__', '__doc__', '__file__', '__loader__', '__name__',
'__package__', '__path__', '__spec__', '_sys', 'boston_housing', 'cifar10', 'cif
ar100', 'fashion_mnist', 'imdb', 'mnist', 'reuters']
'''
#Test 7.4.2020+

#python -m pip install numba
from numba import cuda 
device = cuda.get_current_device()
device.reset()

#import time 


print("1")

cpu = True #False
cpu = False

if cpu:
  #my_devices = #tf.config.experimental.list_physical_devices(device_type='CPU')
  #tf.config.experimental.set_visible_devices(devices= my_devices, device_type='CPU')
  #for anyone who is using tf 2.1, the above comment does not seems to work.
  tf.config.set_visible_devices([], 'GPU')

tf.debugging.set_log_device_placement(True)

mnist = tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0
print(x_train, x_test)

print("2 model =")
model = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(input_shape=(28, 28)),
  tf.keras.layers.Dense(128, activation='relu'),
  tf.keras.layers.Dropout(0.2),
  tf.keras.layers.Dense(10)
])

print("pre predictions = ...")
predictions = model(x_train[:1]).numpy()
print("3")

print(predictions)

tf.nn.softmax(predictions).numpy()

loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

loss_fn(y_train[:1], predictions).numpy()


model.compile(optimizer='adam',
              loss=loss_fn,
              metrics=['accuracy'])

model.fit(x_train, y_train, epochs=5)

model.evaluate(x_test,  y_test, verbose=2)

probability_model = tf.keras.Sequential([
  model,
  tf.keras.layers.Softmax()
])


print(probability_model(x_test[:5]))

device = cuda.get_current_device()
device.reset()
