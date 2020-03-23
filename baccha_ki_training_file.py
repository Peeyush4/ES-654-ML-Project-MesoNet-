import numpy as np
from classifiers import *
from pipeline_3 import *

from keras.preprocessing.image import ImageDataGenerator
import json

def store_predictions(predictions):
    json.dump(predictions,open("predictions.json","w"))

TRAIN_FACE_DICT_FILE_NAME="train_face_dict.json"
TEST_FACE_DICT_FILE_NAME="test_face_dict.json"
TRAIN_FACES_FOLDER="train_face_images"
TEST_FACES_FOLDER="test_face_images"

# 1 - Load the model and its pretrained weights
classifier = Meso4()
print("Doing")

classifier= generate_model(classifier,TRAIN_FACES_FOLDER,TRAIN_FACE_DICT_FILE_NAME,batch_size=50)
classifier.model.save_weights("gazab_baccha.h5")

predictions = compute_accuracy(classifier,TEST_FACES_FOLDER, TEST_FACE_DICT_FILE_NAME)
store_predictions(predictions)
for video_name in predictions:
    print('`{}` video class prediction :'.format(video_name), predictions[video_name][0])
