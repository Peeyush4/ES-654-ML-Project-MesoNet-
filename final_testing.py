import numpy as np
import json,threading
from tqdm import tqdm
from os import listdir,remove,mkdir
from os.path import isfile,join,isdir
from facenet_pytorch import MTCNN
import imageio,torch
from time import time
from math import floor
import cv2 as cv
from scipy.ndimage.interpolation import zoom
from classifiers import *
from pipeline_3 import FaceBatchGenerator,acc_and_logloss_videowise,roundof,accuracy_score,logloss_multiple
PRED_FACE_DICT="pred_dict.json"
PRED_IMAGES="pred_images/"

if not isdir(PRED_IMAGES):
    mkdir(PRED_IMAGES)

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
mtcnn=MTCNN(keep_all=True,post_process=False,device=device)

def store_predictions(predictions):
    json.dump(predictions,open("predictions.json","w"))


def resize_patch(patch,target_size=256):
    patch=patch.permute(1,2,0)
    m, n = patch.shape[:2]
    return zoom(patch.numpy(), (target_size / m, target_size / n, 1))

def store_images(dictionary):

    for image_name in dictionary:
        cv.imwrite(PRED_IMAGES+image_name+".jpg",resize_patch(dictionary[image_name][0]))

def store_json(filename,dictionary):

    if not isfile(filename):
        json.dump(dictionary,open(filename,"w"))
    else:
        original_dictionary=json.load(open(filename,"r"))
        original_dictionary.update(dictionary)
        json.dump(original_dictionary,open(filename,"w"))

class Video:
    def __init__(self, path):
        self.path = path
        self.container = imageio.get_reader(path, 'ffmpeg')
        self.length = self.container.count_frames()
        self.fps = self.container.get_meta_data()['fps']
        self.images={}
        self.faces={}
    
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

def generate_faces_and_facedict(dirname,meta_data_file=None,frame_subsample_count=30,batch_size=2):
    
    if isdir(PRED_FACE_DICT):
        remove(PRED_FACE_DICT) # To remove previous versions

    filenames = [f for f in listdir(dirname) if isfile(join(dirname, f)) and ((f[-4:] == '.mp4') or (f[-4:] == '.avi') or (f[-4:] == '.mov'))]
    
    if meta_data_file is not None:
        meta_data=json.load(open(meta_data_file,"r"))
    else:
        meta_data=None
    thread_list=[]
    for vid in tqdm(filenames):

        print('Dealing with video ', vid)
        if meta_data is not None:
            face_label=meta_data[vid]['label']
        else:
            face_label="not_present"
        video=Video(join(dirname, vid))
        skipstep = max(floor(video.length / frame_subsample_count), 0)
        finder_frameset = range(0, video.length, skipstep + 1)

        face_counter=0
        a=time()
        for lb in np.arange(0, len(finder_frameset), batch_size):
            imgs = [video.get(i) for i in finder_frameset[lb:lb+batch_size]]
            multiple_frame_faces=mtcnn(imgs)
            for frame_faces in multiple_frame_faces:
                if frame_faces is None:
                    continue
                if type(frame_faces)==list:    
                    if len(frame_faces)==0:
                        continue
                    for face in frame_faces:
                        if face is None:
                            continue
                        video.faces[vid[:-4]+"_"+str(face_counter)]=face_label
                        video.images[vid[:-4]+"_"+str(face_counter)]=face
                        face_counter+=1
                else:
                    video.faces[vid[:-4]+"_"+str(face_counter)]=face_label
                    video.images[vid[:-4]+"_"+str(face_counter)]=frame_faces
                    face_counter+=1
        b=time()
        store_json(PRED_FACE_DICT,video.faces)
        p1=threading.Thread(target=store_images,args=(video.images,))
        p1.start()
        thread_list.append(p1)
        # store_images(video.images)
        c=time()
        print("Time taken for finding all faces: ",b-a," Time taken for storing all faces: ",c-b)
    for thread in thread_list:
        thread.join()
    print("All threads joined")

def prb2label(pred):
    if pred<=0.5:
        return "REAL"
    else:
        return "FAKE"

def compute_accuracy(classifier, test_image_dir, face_dict, frame_subsample_count = 30,batch_size=50):
    '''
    Extraction + Prediction over a video
    '''
    face_dict=json.load(open(face_dict,"r"))
    gen = FaceBatchGenerator(face_dict,test_image_dir)
    all_actual=[]
    all_pred=[]
    all_face_names=[]

    for _ in tqdm(range(gen.length // batch_size + 1)):

        face_batch,actual_label = gen.next_batch(batch_size = batch_size)
        face_names = gen.batch_names
        prediction = classifier.predict(face_batch)        
        all_actual.extend(actual_label)
        all_pred.extend(prediction)
        all_face_names.extend(face_names)
        
        if actual_label[0]!="not_present":
            print("Accuracy till now: ",accuracy_score(all_actual,roundof(all_pred)))
            print("Log loss till now: ",logloss_multiple(all_actual,all_pred))

            acc,logl=acc_and_logloss_videowise(all_face_names,all_actual,all_pred)
            print("Accuracy Videowise: ",acc)
            print("Logloss Videowise: ",logl)
    
        predictions=predictions_from_face_results(all_face_names,all_pred)
        for name in predictions:
            predictions[name]={"probability":predictions[name],"prediction":prb2label(predictions[name])}
    return predictions

def predictions_from_face_results(all_face_names,pred_labels):
    face_name_dict=dict()

    for face_name,pred_label in zip(all_face_names,pred_labels):
        video_name=face_name.split("_")[0]
        if video_name in face_name_dict:
            face_name_dict[video_name].append(pred_label)
        else:
            face_name_dict[video_name]=[pred_label]

    for video_name in face_name_dict:
        pred_labels=face_name_dict[video_name]
        pred=np.mean(np.array(pred_labels)>0.5)
        face_name_dict[video_name]=pred

    return face_name_dict

PREDICTION_FOLDER="prediction_videos/"
MODEL_FILE="models/gazab_baccha.h5"
METADATA_FILE="metadata.json"
# METADATA_FILE=None
generate_faces_and_facedict(PREDICTION_FOLDER,METADATA_FILE,batch_size=5)
torch.cuda.device(None)
classifier=Meso4()
classifier.load(MODEL_FILE)
predictions=compute_accuracy(classifier,PRED_IMAGES,PRED_FACE_DICT)
store_predictions(predictions)