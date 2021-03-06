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

FACE_DICT_FILE_NAME="train_face_dict.json"
FINAL_IMAGES_FOLDER="train_face_images/"

work_done_previously=True

if not isdir(FINAL_IMAGES_FOLDER):
    work_done_previously=False
    mkdir(FINAL_IMAGES_FOLDER)

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
mtcnn=MTCNN(keep_all=True,post_process=False,device=device)

def resize_patch(patch,target_size=256):
    patch=patch.permute(1,2,0)
    m, n = patch.shape[:2]
    return zoom(patch.numpy(), (target_size / m, target_size / n, 1))

def store_images(dictionary):

    for image_name in dictionary:
        cv.imwrite(FINAL_IMAGES_FOLDER+image_name+".jpg",resize_patch(dictionary[image_name][0]))

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
        # self.fps = self.container.get_meta_data()['fps']
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

def filter_facedict(facedict,worked_filenames):
    original_dictionary=json.load(open(facedict,"r"))
    for key in original_dictionary:
        if key.split("_")+".mp4" in worked_filenames:
            pass
        else:
            print("not there")
            # del original_dictionary[key]
    json.dump(original_dictionary,open(facedict,"w"))

def generate_faces_and_facedict(dirname,meta_data_file,frame_subsample_count=30,batch_size=2):
    
    # remove(FACE_DICT_FILE_NAME) # To remove previous versions

    all_filenames = [f for f in listdir(dirname) if isfile(join(dirname, f)) and (f[-4:] == '.mp4')]
    
    if not work_done_previously:
        filenames=all_filenames
    elif not isfile(FACE_DICT_FILE_NAME):
        filenames=all_filenames
    else:
        files_worked_previously=listdir(FINAL_IMAGES_FOLDER)
        videos_worked_previously=list(set([filename.split("_")[0]+".mp4" for filename in files_worked_previously]))
        print("Previously worked videos: ",len(videos_worked_previously))
        filenames=all_filenames-videos_worked_previously
        filter_facedict(FACE_DICT_FILE_NAME,filenames)


    meta_data=json.load(open(meta_data_file,"r"))
    thread_list=[]
    for vid in tqdm(filenames):

        # print('Dealing with video ', vid)
        face_label=meta_data[vid]['label']
        video=Video(join(dirname, vid))
        skipstep = max(floor(video.length / frame_subsample_count), 0)
        finder_frameset = range(0, video.length, skipstep + 1)

        face_counter=0
        # a=time()
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
        # b=time()
        store_json(FACE_DICT_FILE_NAME,video.faces)
        p1=threading.Thread(target=store_images,args=(video.images,))
        p1.start()
        del video
        thread_list.append(p1)
        # store_images(video.images)
        # c=time()
        # print("Time taken for finding all faces: ",b-a," Time taken for storing all faces: ",c-b)
    for thread in thread_list:
        thread.join()
generate_faces_and_facedict("train_videos","all_metadata.json",batch_size=10)