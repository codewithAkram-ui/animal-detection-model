import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
import json


model_path = "F:/animals/new/wild_animal_detector.h5"# paste your model path here example "F:\animals\new\wild_animal_detector.h5"
model = tf.keras.models.load_model(model_path)
with open("F:/animals/new/wild_animal_detector.json", 'r') as f:# paste your file here example "F:\animals\new\wild_animal_detector.json"
    class_labels = json.load(f)
test_image_path ="C:/Users/sk764/Downloads/3.WEBP" # upload images for test here
img = load_img(test_image_path, target_size=(128, 128))
img_array = img_to_array(img) / 255.0  
img_array = np.expand_dims(img_array, axis=0)  
predictions = model.predict(img_array)
predicted_class_index = np.argmax(predictions[0])
predicted_class_label = class_labels[str(predicted_class_index)]

print(f"Predicted class: {predicted_class_label}")
# animal {"0": "Chimpanzee", "1": "antelope", "2": "badger", "3": "bear", "4": "boar", "5": "buffalo", "6": "cheetah", "7": "coyote", "8": "deer", "9": "elephant", "10": "fox", "11": "giraffe", "12": "gorilla", "13": "hippopotamus", "14": "kangaroo", "15": "koala", "16": "leopard", "17": "lion", "18": "okapi", "19": "orangutan", "20": "otter", "21": "ox", "22": "panda", "23": "porcupine", "24": "raccoon", "25": "reindeer", "26": "rhino", "27": "seal", "28": "shark", "29": "tiger", "30": "turtle", "31": "whale", "32": "wolf", "33": "zebra"}