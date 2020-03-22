import numpy as np
from classifiers import *
from pipeline_2 import *

from keras.preprocessing.image import ImageDataGenerator
import json
# def combine_all_metadata_files(metadata_folder):
def store_predictions(predictions):
    json.dump(predictions,open("predictions.json","w"))


# 1 - Load the model and its pretrained weights
classifier = Meso4()
# classifier.load('weights/Meso4_DF')
dirname="train_videos"
meta_data_file="metadata.json"
print("Doing")
classifier= generate_model(classifier,dirname,meta_data_file,batch_size=20)
classifier.model.save_weights("gazab.h5")

predictions = compute_accuracy(classifier, dirname, meta_data_file)
store_predictions(predictions)
for video_name in predictions:
    print('`{}` video class prediction :'.format(video_name), predictions[video_name][0])
