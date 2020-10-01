import tensorflow.compat.v1 as tf
from imageai.Detection import ObjectDetection
import os

'''
ImageIO example

Setup: E:\ - RAM disk, Imdisk
Python 3.7.9
Download YOLO: https://github.com/OlafenwaMoses/ImageAI/releases/tag/1.0/
python -m pip install tensorflow==1.15.0  
python -m pip install keras==2.2.5

ImageAI doesn't support TF 2.0 and TF 1.15 or lower requires Keras up to 2.2.5 or older version
These settings invoke "deprecated" warning, maybe even older would be better.
'''

execution_path = os.getcwd()
yolo_path = "e:\\yolo.h5"
localdir = False

detector = ObjectDetection()
detector.setModelTypeAsYOLOv3()
if localdir:
    detector.setModelPath( os.path.join(execution_path , "yolo.h5"))
else: 
    detector.setModelPath( yolo_path)

#dir(detector)
detector.loadModel()
#loaded_model = tf.keras.models.load_model("./src/mood-saved-models/"model + ".h5")
#loaded_model = tf.keras.models.load_model(detector.)

path = "E:\capture_023_29092020_150305.jpg" #IMG_20200528_044908.jpg"
pathOut = "E:\YOLO_capture_023_29092020_150305.jpg"

path = "E:\\capture_046_29092020_150628.jpg"
pathOut = "E:\\yolo_out.jpg"

detections = detector.detectObjectsFromImage(input_image=os.path.join(execution_path , path), output_image_path=os.path.join(execution_path , pathOut), minimum_percentage_probability=30)

for eachObject in detections:
    print(eachObject["name"] , " : ", eachObject["percentage_probability"], " : ", eachObject["box_points"] )
    print("--------------------------------")
    
