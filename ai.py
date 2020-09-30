import tensorflow.compat.v1 as tf
from imageai.Detection import ObjectDetection
import os

execution_path = os.getcwd()

detector = ObjectDetection()
detector.setModelTypeAsYOLOv3()
detector.setModelPath( os.path.join(execution_path , "yolo.h5"))

#dir(detector)
detector.loadModel()
#loaded_model = tf.keras.models.load_model("./src/mood-saved-models/"model + ".h5")
#loaded_model = tf.keras.models.load_model(detector.)

path = "IMG_20200528_044908.jpg"
pathOut = "IMG_20200528_044908_NEW.jpg"
detections = detector.detectObjectsFromImage(input_image=os.path.join(execution_path , path), output_image_path=os.path.join(execution_path , pathOut), minimum_percentage_probability=30)

for eachObject in detections:
    print(eachObject["name"] , " : ", eachObject["percentage_probability"], " : ", eachObject["box_points"] )
    print("--------------------------------")