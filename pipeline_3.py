import json
from os import listdir
from os.path import isfile, join
import cv2 as cv
import numpy as np
from tqdm import tqdm
from sklearn.metrics import accuracy_score
from random import shuffle
## Face prediction

class FaceBatchGenerator:
    '''
    Made to deal with framesubsets of video.
    '''
    def __init__(self, face_dict, image_dir, target_size = 256):
        self.face_dict = face_dict
        self.image_name_list=list(self.face_dict.keys())
        self.target_size = target_size
        self.head = 0
        self.length = len(face_dict)
        self.image_dir=image_dir
    def shuffle(self):
        shuffle(self.image_name_list)
        self.head=0
    def get_face_and_label(self,i):
        image_name=self.image_name_list[i]+".jpg"
        label=self.face_dict[self.image_name_list[i]]
        if label=="not_present":
            pass
        elif label=="FAKE":
            label=1
        else:
            label=0
        return cv.imread(self.image_dir+image_name),label
    
    def next_batch(self, batch_size = 50):
        batch = np.zeros((1, self.target_size, self.target_size, 3))
        # stop = min(self.head + batch_size, self.length)
        self.train_labels=[]
        self.batch_names=[]
        i = 0
        while (i < batch_size) and (self.head < self.length):
            patch,label = self.get_face_and_label(self.head)
            self.batch_names.append(self.image_name_list[self.head])
            batch =np.concatenate((batch, np.expand_dims(patch, axis = 0)),
                                        axis = 0)
            i += 1
            self.head += 1
            self.train_labels.append(label)
            
        return batch[1:],self.train_labels


def predict_faces(generator, classifier, batch_size = 50, output_size = 1):
    '''
    Compute predictions for a face batch generator
    '''
        
    for epoch in range(generator.length // batch_size + 1):
        face_batch = generator.next_batch(batch_size = batch_size)
        prediction = classifier.predict(face_batch )
        if (len(prediction) > 0):
            profile = np.concatenate((profile, prediction))
    return profile[1:]

def train_faces(generator, classifier, epochs=10,batch_size = 50, output_size = 1):
    '''
    Train for a face batch generator
    '''
    print("Total epochs: ",generator.length//batch_size+1)
    for epoch in tqdm(range(epochs)):
        print("Training epoch:",epoch+1)
        generator.shuffle()
        for _ in tqdm(range(generator.length // batch_size + 1)):
       
            face_batch,train_labels = generator.next_batch(batch_size = batch_size)
            if len(face_batch):
                classifier.fit(face_batch, train_labels)
                #prediction=classifier.predict(face_batch)
                #print("Training accuracy on this batch: ",accuracy_score(train_labels,roundof(prediction)))
        classifier.model.save_weights("gazab_baccha_"+str(epoch)+".h5")
    return classifier

def generate_model(classifier, train_images_dir,face_dict, frame_subsample_count = 30,batch_size=50,epochs=4):
    face_dict=json.load(open(face_dict,"r"))
    gen = FaceBatchGenerator(face_dict,train_images_dir)
    classifier= train_faces(gen, classifier, batch_size=batch_size,epochs=epochs)
    return classifier

def logloss(actual_label,pred_label):
    x=actual_label*np.log(pred_label+1e-6)+(1-actual_label)*np.log(1-pred_label+1e-6)
    return min(1,-x)

def roundof(lis):
    try:
        return [round(float(i[0])) for i in lis]
    except:
        return [round(float(i)) for i in lis]

def logloss_multiple(actual_labels,pred_labels):
    return (1/len(actual_labels))*sum([logloss(actual_label,pred_label) for (actual_label,pred_label) in zip(actual_labels,pred_labels)])

def acc_and_logloss_videowise(all_face_names,actual_labels,pred_labels):

    face_name_dict=dict()

    for face_name,actual_label,pred_label in zip(all_face_names,actual_labels,pred_labels):
        video_name=face_name.split("_")[0]
        if video_name in face_name_dict:
            face_name_dict[video_name][0].append(actual_label)
            face_name_dict[video_name][1].append(pred_label)
        else:
            face_name_dict[video_name]=[[actual_label],[pred_label]]

    all_actual=[]
    all_pred=[]

    for video_name in face_name_dict:
        pred_labels=face_name_dict[video_name][1]
        actual_labels=face_name_dict[video_name][0]
        pred=np.mean(np.array(pred_labels)>0.5)
        all_actual.append(actual_labels[0])
        all_pred.append(pred)
    
    return accuracy_score(roundof(all_pred),all_actual),logloss_multiple(all_actual,all_pred) 

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
    # predictions = {}
    all_actual=[]
    all_pred=[]
    all_face_names=[]

    for epoch in range(gen.length // batch_size + 1):
        face_batch,actual_label = gen.next_batch(batch_size = batch_size)
        face_names = gen.batch_names
        prediction = classifier.predict(face_batch)
        # for name,pred in zip(face_names,prediction):
        #     predictions[name]=pred
        all_actual.extend(actual_label)
        all_pred.extend(prediction)
        all_face_names.extend(face_names)
        print("Accuracy till now: ",accuracy_score(all_actual,roundof(all_pred)))
        print("Log loss till now: ",logloss_multiple(all_actual,all_pred))

        acc,logl=acc_and_logloss_videowise(all_face_names,all_actual,all_pred)
        print("Accuracy Videowise: ",acc)
        print("Logloss Videowise: ",logl)

    predictions=predictions_from_face_results(all_face_names,all_pred)
    for name in predictions:
        predictions[name]={"probability":predictions[name],"prediction":prb2label(predictions[name])}
    return predictions
