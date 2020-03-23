# -*- coding:utf-8 -*-

import random, json,torch
from os import listdir
from os.path import isfile, join
import cv2 as cv
import numpy as np
from math import floor
from scipy.ndimage.interpolation import zoom, rotate

import imageio
# import face_recognition
from facenet_pytorch import MTCNN
# from mtcnn.mtcnn import MTCNN
from tqdm import tqdm

#device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
device=None
mtcnn=MTCNN(keep_all=False,post_process=False,device=device)
## Face extraction

class Video:
    def __init__(self, path):
        self.path = path
        self.container = imageio.get_reader(path, 'ffmpeg')
        self.length = self.container.count_frames()
        self.fps = self.container.get_meta_data()['fps']
    
    def init_head(self):
        self.container.set_image_index(0)
    
    def next_frame(self):
        self.container.get_next_data()
    
    def get(self, key):
        return self.container.get_data(key)
    
    def __call__(self, key):
        return self.get(key)
    
    def __len__(self):
        return self.length


class FaceFinder(Video):
    def __init__(self, path, load_first_face = True):
        super().__init__(path)
        self.faces = []
        self.coordinates = {}  # stores the face (locations center, rotation, length)
        self.last_frame = self.get(0)
        self.frame_shape = self.last_frame.shape[:2]
        self.last_location = (0, 200, 200, 0)

    def load_coordinates(self, filename):
        np_coords = np.load(filename)
        self.coordinates = np_coords.item()
    
    def expand_location_zone(self, loc, margin = 0.2):
        ''' Adds a margin around a frame slice '''
        offset = round(margin * (loc[2] - loc[0]))
        y0 = max(loc[0] - offset, 0)
        x1 = min(loc[1] + offset, self.frame_shape[1])
        y1 = min(loc[2] + offset, self.frame_shape[0])
        x0 = max(loc[3] - offset, 0)
        return (y0, x1, y1, x0)

    
    @staticmethod
    def upsample_location(reduced_location, upsampled_origin, factor):
        ''' Adapt a location to an upsampled image slice '''
        y0, x1, y1, x0 = reduced_location
        Y0 = round(upsampled_origin[0] + y0 * factor)
        X1 = round(upsampled_origin[1] + x1 * factor)
        Y1 = round(upsampled_origin[0] + y1 * factor)
        X0 = round(upsampled_origin[1] + x0 * factor)
        return (Y0, X1, Y1, X0)

    @staticmethod
    def pop_largest_location(location_list):
        max_location = location_list[0]
        max_size = 0
        if len(location_list) > 1:
            for location in location_list:
                size = location[2] - location[0]
                if size > max_size:
                    max_size = size
                    max_location = location
        return max_location
    
    @staticmethod
    def L2(A, B):
        return np.sqrt(np.sum(np.square(A - B)))
    
    def find_faces(self, resize = 0.5, stop = 0, skipstep = 0, no_face_acceleration_threshold = 3, cut_left = 0, cut_right = -1, use_frameset = False, frameset = []):
        '''
        The core function to extract faces from frames
        using previous frame location and downsampling to accelerate the loop.
        '''
        
        # to only deal with a subset of a video, for instance I-frames only
        if (use_frameset):
            finder_frameset = frameset
        else:
            if (stop != 0):
                finder_frameset = range(0, min(self.length, stop), skipstep + 1)
            else:
                finder_frameset = range(0, self.length, skipstep + 1)

        # all_frames=[self.get(i) for i in finder_frameset]
        batch_size=2
        for lb in np.arange(0, len(finder_frameset), batch_size):
            imgs = [self.get(i) for i in finder_frameset[lb:lb+batch_size]]
            self.faces.extend(mtcnn(imgs))
#            print(len(mtcnn(imgs)),mtcnn(imgs)[0].shape) 
 
    def get_face(self, i):
        ''' Basic unused face extraction without alignment '''
        frame = self.faces[i]
 #       print(frame.shape)
        return frame


## Face prediction

class FaceBatchGenerator:
    '''
    Made to deal with framesubsets of video.
    '''
    def __init__(self, face_finder, target_size = 256):
        self.finder = face_finder
        self.target_size = target_size
        self.head = 0
        self.length = int(face_finder.length)

    def resize_patch(self, patch):
        patch=patch.permute(1,2,0)
        m, n = patch.shape[:2]
#        return cv.resize(patch,(self.target_size,self.target_size))
        return zoom(patch.numpy(), (self.target_size / m, self.target_size / n, 1))
    
    def next_batch(self, batch_size = 50):
        batch = np.zeros((1, self.target_size, self.target_size, 3))
        stop = min(self.head + batch_size, self.length)
        i = 0
        while (i < batch_size) and (self.head < len(self.finder.faces)):
            # if self.head in self.finder.coordinates:
            patch = self.finder.get_face(self.head)
            i += 1
            self.head += 1
            if patch is None:
                print("None")
                continue
            if patch is []:
                print("empty")
                continue
 #           print(patch.shape)
#            if type(patch)==list:
 #               patch=patch[0]
  #          print(self.resize_patch(patch.numpy()).shape)
            batch =np.concatenate((batch, np.expand_dims(self.resize_patch(patch), axis = 0)),
                                        axis = 0)
            
        return batch[1:]


def predict_faces(generator, classifier, batch_size = 50, output_size = 1):
    '''
    Compute predictions for a face batch generator
    '''
        
    n = len(generator.finder.faces)
    profile = np.zeros((1, output_size))
    for epoch in range(n // batch_size + 1):
        face_batch = generator.next_batch(batch_size = batch_size)
        prediction = classifier.predict(face_batch )
        if (len(prediction) > 0):
            profile = np.concatenate((profile, prediction))
    return profile[1:]

def train_faces(generator, classifier, train_labels, batch_size = 50, output_size = 1):
    '''
    Train for a face batch generator
    '''
    n = len(generator.finder.faces)
    print("n faces: ",n)
    if n==0:
        return classifier
    print(n // batch_size + 1)
    for epoch in range(n // batch_size + 1):
        # print("Training on ",epoch," epoch")
        face_batch = generator.next_batch(batch_size = batch_size)
        # print(len(face_batch))
        if len(face_batch):
            classifier.fit(face_batch, train_labels[:len(face_batch)])
    return classifier

def generate_model(classifier, dirname,meta_data_file, frame_subsample_count = 30,batch_size=50):
    filenames = [f for f in listdir(dirname) if isfile(join(dirname, f)) and ((f[-4:] == '.mp4') or (f[-4:] == '.avi') or (f[-4:] == '.mov'))]
    meta_data=json.load(open(meta_data_file,"r"))
    for vid in tqdm(filenames[:2000]):
        print('Dealing with video ', vid)
        if meta_data[vid]['label']=='FAKE':
            train_labels=[1]*batch_size
        else:
            train_labels=[0]*batch_size
        
        # Compute face locations and store them in the face finder
        face_finder = FaceFinder(join(dirname, vid), load_first_face = False)
        skipstep = max(floor(face_finder.length / frame_subsample_count), 0)
        face_finder.find_faces(resize=0.5, skipstep = skipstep)
        
        # print('Training ', vid)
        gen = FaceBatchGenerator(face_finder)
        classifier= train_faces(gen, classifier, train_labels, batch_size=batch_size)
    return classifier

def logloss(actual_label,pred_label):
    x=actual_label*np.log(pred_label+1e-6)+(1-actual_label)*np.log(1-pred_label+1e-6)
    return min(1,-x)

def compute_accuracy(classifier, dirname,meta_data_file, frame_subsample_count = 30):
    '''
    Extraction + Prediction over a video
    '''
    filenames = [f for f in listdir(dirname) if isfile(join(dirname, f)) and ((f[-4:] == '.mp4') or (f[-4:] == '.avi') or (f[-4:] == '.mov'))]
    predictions = {}
    accuracy_sum=0
    logloss_sum=0
    counter=0
    meta_data=json.load(open(meta_data_file,"r"))

    for vid in tqdm(filenames):
        print('Dealing with video ', vid)
        
        # Compute face locations and store them in the face finder
        face_finder = FaceFinder(join(dirname, vid), load_first_face = False)
        skipstep = max(floor(face_finder.length / frame_subsample_count), 0)
        face_finder.find_faces(resize=0.5, skipstep = skipstep)
        # print(len(face_finder))
        print('Predicting ', vid)
        gen = FaceBatchGenerator(face_finder)
        p = predict_faces(gen, classifier)
        
        predictions[vid[:-4]] = (np.mean(p > 0.5), p)
        if meta_data[vid]['label']=='FAKE':
            actual_label=1
        else:
            actual_label=0
        
        if round(np.mean(p>0.5))==actual_label:
            accuracy_sum+=1
            print("Accurate")
        else:
            print("Not accurate")

        print("Actual: ",actual_label,"Predicted: ",np.mean(p > 0.5))
        print("Logloss in this video: ",logloss(actual_label,np.mean(p > 0.5)))
        counter+=1
        logloss_sum+=logloss(actual_label,np.mean(p > 0.5))
        average_logloss=logloss_sum/counter
        print("Average logloss till now: ",average_logloss)
        average_accuracy=accuracy_sum/counter
        print("Average accuracy till now: ",average_accuracy)

    return predictions
