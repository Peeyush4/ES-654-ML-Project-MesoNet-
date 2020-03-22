import numpy as np
from classifiers import *
from pipeline_2 import *

from keras.preprocessing.image import ImageDataGenerator

# 1 - Load the model and its pretrained weights
classifier = Meso4()
# classifier.load('weights/Meso4_DF')

# # 2 - Minimial image generator
# # We did use it to read and compute the prediction by batchs on test videos
# # but do as you please, the models were trained on 256x256 images in [0,1]^(n*n)

# dataGenerator = ImageDataGenerator(rescale=1./255)
# generator = dataGenerator.flow_from_directory(
#         'test_images',
#         target_size=(256, 256),
#         batch_size=1,
#         class_mode='binary',
#         subset='training')

# # 3 - Predict
# X, y = generator.next()
# print('Predicted :', classifier.predict(X), '\nReal class :', y)

# 4 - Prediction for a video dataset
import json
meta_data=json.load(open("metadata.json","r"))

classifier.load('gazab.h5')

predictions = compute_accuracy(classifier, 'test_videos')
for video_name in predictions:
        if meta_data[video_name+".mp4"]['label']=='FAKE':
                print("Fake video")
        else:
                print("Real video")
                
        print('`{}` video class prediction :'.format(video_name), predictions[video_name][0])