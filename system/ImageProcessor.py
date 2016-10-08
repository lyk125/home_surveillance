# ImageProcessor.
# Brandon Joffe
# 2016
#
# Copyright 2016, Brandon Joffe, All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Code used in this project included opensource software (openface)
# developed by Brandon Amos
# Copyright 2015-2016 Carnegie Mellon University

import cv2
import numpy as np
import os
import glob
#from skimage import io
import dlib
import sys
import argparse
import imagehash
import json
from PIL import Image
import urllib
import base64
import pickle
import math
import datetime
#import imutils
import threading
import logging


from sklearn.decomposition import PCA
from sklearn.grid_search import GridSearchCV
from sklearn.manifold import TSNE
from sklearn.svm import SVC

import time
start = time.time()
from operator import itemgetter

import numpy as np
np.set_printoptions(precision=2)
import pandas as pd

import Camera
import openface

fileDir = os.path.dirname(os.path.realpath(__file__))
modelDir = os.path.join(fileDir, '..', 'models')
dlibModelDir = os.path.join(modelDir, 'dlib')
openfaceModelDir = os.path.join(modelDir, 'openface')

parser = argparse.ArgumentParser()
parser.add_argument('--dlibFacePredictor', type=str, help="Path to dlib's face predictor.",
                    default=os.path.join(dlibModelDir, "shape_predictor_68_face_landmarks.dat"))
parser.add_argument('--networkModel', type=str, help="Path to Torch network model.",
                    default=os.path.join(openfaceModelDir, 'nn4.small2.v1.t7'))
parser.add_argument('--imgDim', type=int,
                    help="Default image dimension.", default=96)
parser.add_argument('--cuda', action='store_true')
parser.add_argument('--unknown', type=bool, default=False,
                    help='Try to predict unknown people')

args = parser.parse_args()

align = openface.AlignDlib(args.dlibFacePredictor)

neuralNet_lock = threading.Lock()

cascade_lock = threading.Lock()

accurate_cascade_lock = threading.Lock()

net = openface.TorchNeuralNet(args.networkModel, imgDim=args.imgDim,cuda=args.cuda) 

facecascade = cv2.CascadeClassifier("cascades/haarcascade_frontalface_alt2.xml")
facecascade2 = cv2.CascadeClassifier("cascades/haarcascade_frontalface_alt2.xml")
uppercascade = cv2.CascadeClassifier("cascades/haarcascade_upperbody.xml")
eyecascade = cv2.CascadeClassifier("cascades/haarcascade_eye.xml")

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(args.dlibFacePredictor)

# with open("generated-embeddings/classifier.pkl", 'r') as f:
#         (le, clf) = pickle.load(f) # loads labels and classifier SVM or GMM

def motion_detector(camera,frame, get_rects):
        #calculate mean standard deviation then determine if motion has actually accurred
        height, width, channels = frame.shape

        text = "Unoccupied"
        occupied = False
        kernel = np.ones((5,5),np.uint8)

        # resize the frame, convert it to grayscale, filter and blur it
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        gray = clahe.apply(gray)
        #gray = cv2.GaussianBlur(gray, (11, 11), 0)
        gray = cv2.medianBlur(gray,9)  # filters out noise
    
        #cv2.imwrite("grayfiltered.jpg", gray)
        people_rects = []

        #initialise and build some history
        if camera.history == 0:
            camera.current_frame = gray
            camera.history +=1
            if get_rects == True: # return peoplerects without frame
                return occupied,  people_rects #people_rects #people_rects  #occupied, 
            else:
                return occupied,  frame
        elif camera.history == 1:
            camera.previous_frame = camera.current_frame
            camera.current_frame = gray
            #camera.next_frame = gray
            camera.meanframe = cv2.addWeighted(camera.previous_frame,0.5,camera.current_frame,0.5,0)
            camera.history +=1
            if get_rects == True: # return peoplerects without frame
                return occupied,  people_rects #people_rects #people_rects  #occupied, 
            else:
                return occupied,  frame
        elif camera.history == 2:
            camera.previous_frame = camera.meanframe
            camera.current_frame = gray
            #camera.next_frame = gray
            camera.meanframe = cv2.addWeighted(camera.previous_frame,0.5,camera.current_frame,0.5,0)
            cv2.imwrite("avegrayfiltered.jpg", camera.meanframe)
            camera.history +=1
            if get_rects == True: # return peoplerects without frame
                return occupied,  people_rects #people_rects #people_rects  #occupied, 
            else:
                return occupied,  frame
        elif camera.history > 2000 and len(camera.trackers) == 0:
            camera.previous_frame = camera.current_frame
            camera.current_frame = gray
            #camera.next_frame = gray
            camera.history = 0

        # compute the absolute difference between the current frame and
        # first frame
        frameDelta = cv2.absdiff(camera.meanframe , gray)
        thresh = cv2.threshold(frameDelta, 25, 255, cv2.THRESH_BINARY)[1]
        

        # dilate the thresholded image to fill in holes, then find contours
        # on thresholded image

        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel) # removes small holes i.e noise
        thresh = cv2.dilate(thresh, kernel, iterations=3) # increases white region by saturating blobs
        cv2.imwrite("motion.jpg", thresh)
        (cnts, _) = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE)
        # loop over the contours

    
        for c in cnts:
            # if the contour is too small or too big, ignore it
            if cv2.contourArea(c) < 1000 or cv2.contourArea(c) > 90000:
                if cv2.contourArea(c) > 100000:
                    camera.history = 0
                    break
                continue
               
            # compute the bounding box for the contour, draw it on the frame,
            # and update the text
           
            (x, y, w, h) = cv2.boundingRect(c)
            if (h-y) == height or (w-x) == width:
                    camera.history = 0
                    break

            if (h) > (w):
                #cv2.rectangle(frame, (x, y ), (x + w, y + h), (0, 255, 0), 2)
               
                text = "Occupied"
                occupied = True
                people_rects.append(cv2.boundingRect(c))
           
        # draw the text and timestamp on the frame
            # cv2.putText(frame, "Room Status: {}".format(text), (10, 10),
            # cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1)
            # cv2.putText(frame, datetime.datetime.now().strftime("%A %d %B %Y %I:%M:%S%p"),
            # (10, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1)
        
        # if len(cnts) > 0:
        #     return True
        # return False

        camera.history +=1

        #motionData = {'occupied':occupied,'frame':frame, 'peopleRects': people_rects}
        #print str(occupied) + "==================>    motion detected\n\n\n\n"
        if get_rects == True: # return peoplerects without frame
            return occupied,  people_rects #people_rects #people_rects  #occupied, 
        else:
            return occupied,  frame
    
def resize(frame):
    r = 420.0 / frame.shape[1]
    dim = (420, int(frame.shape[0] * r))
    # perform the actual resizing of the image and show it
    frame = cv2.resize(frame, dim, interpolation = cv2.INTER_AREA)    
    return frame  

def crop(image, box, dlibRect = False):

    if dlibRect == False:
       x, y, w, h = box
       return image[y: y + h, x: x + w] 

    return image[box.top():box.bottom(), box.left():box.right()]

def is_inside(o, i):
    ox, oy, ow, oh = o
    ix, iy, iw, ih = i
    return ox > ix and oy > iy and ox + ow < ix + iw and oy + oh < iy + ih

def draw_boxes(image, rects, dlibrects):
   
   if dlibrects:
       image = draw_rects_dlib(image, rects)
   else:
       image = draw_rects_cv(image, rects)

   return image


def draw_rects_cv(img, rects, color=(0, 40, 255)):

    overlay = img.copy()
    output = img.copy()
    
    count = 1
    for x, y, w, h in rects:
      
        cv2.rectangle(overlay, (x, y), (x+w, y+h), color, 2)
        cv2.addWeighted(overlay, 0.5, output, 0.5, 0, output)
    return output
   
def draw_rects_dlib(img, rects, color = (0, 255, 255)):

    overlay = img.copy()
    output = img.copy()
      
    for bb in rects:
        bl = (bb.left(), bb.bottom()) # (x, y)
        tr = (bb.right(), bb.top()) # (x+w,y+h)
        cv2.rectangle(overlay, bl, tr, color, thickness=2)
        cv2.addWeighted(overlay, 0.5, output, 0.5, 0, output)       
    return output

def draw_text(image, persondict):
    cv2.putText(image,  str(persondict['name']) + " " + str(math.ceil(persondict['confidence']*100))+ "%", (bb.left()-15, bb.bottom() + 10),
                    cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.25,
                    color=(152, 255, 204), thickness=1)

def draw_rect(img,x,y,w,h, color=(0, 40, 255)):

    overlay = img.copy()
    output = img.copy()
          
    cv2.rectangle(overlay, (x, y), (x+w, y+h), color, 2)
    cv2.addWeighted(overlay, 0.5, output, 0.5, 0, output)

    return output

def draw_rects_dlib(img, rects):

    overlay = img.copy()
    output = img.copy()
      
    for bb in rects:
        bl = (bb.left(), bb.bottom()) # (x, y)
        tr = (bb.right(), bb.top()) # (x+w,y+h)
        cv2.rectangle(overlay, bl, tr, color=(0, 255, 255), thickness=2)
        cv2.addWeighted(overlay, 0.5, output, 0.5, 0, output)       
    return output


def pre_processing(image):
     
     grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
     #equ = cv2.equalizeHist(image)
     clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
     cl1 = clahe.apply(grey)
     cv2.imwrite('clahe_2.jpg',cl1)

     return cl1


def rgb_pre_processing(image):

    (h, w) = image.shape[:2]    
    zeros = np.zeros((h, w), dtype="uint8")
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(10,10))
    (B, G, R) = cv2.split(image)
    R = clahe.apply(R)
    G = clahe.apply(G)
    B = clahe.apply(B)
 
    filtered = cv2.merge([B, G, R])
    cv2.imwrite('notfilteredRGB.jpg',image)
    cv2.imwrite('filteredRGB.jpg',filtered)
    return filtered

def detect_faces(image,dlibdetector):

     if dlibdetector:
        #image = cv2.flip(image, 1)
        return detectdlib_face_rgb(image)
     else:
        return detect_cascadeface(image)


def detectdlib_face_rgb(rgbFrame):
    #rgbFrame = rgb_pre_processing(rgbFrame)
    bbs = detector(rgbFrame, 1)

    return bbs#, annotatedFrame

def detect_cascadeface(image):
   
    #image = rgb_pre_processing(image)
    with cascade_lock:  # used to block simultaneous access to resource, stops segmentation fault when using more than one camera
        #logging.debug('\n\n////////////////////// 1.1 //////////////////////\n\n')  
        image = pre_processing(image)
        rects = facecascade.detectMultiScale(image, scaleFactor=1.2, minNeighbors=4, minSize=(30, 30), flags = cv2.CASCADE_SCALE_IMAGE)
        #logging.debug('\n\n////////////////////// 1.2 //////////////////////\n\n')  
    return rects

def detect_cascadeface_accurate(img):
    #img = pre_processing(img)
    with accurate_cascade_lock:
        rects = facecascade2.detectMultiScale(img, scaleFactor=1.05, minNeighbors=5, minSize=(30, 30), flags = cv2.CASCADE_SCALE_IMAGE)
    return rects

def detect_cascade(img, cascade):
    rects = cascade.detectMultiScale(img, scaleFactor=1.2, minNeighbors=4, minSize=(40, 40), flags = cv2.CASCADE_SCALE_IMAGE)
    return rects

def detect_people_hog(image):
    image = rgb_pre_processing(image)
    hog = cv2.HOGDescriptor()
    hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
    found, w = hog.detectMultiScale(image,winStride=(30,30),padding=(16,16), scale=1.1)

    filtered_detections = []
  
    for ri, r in enumerate(found):
        for qi, q in enumerate(found):
            if ri != qi and is_inside(r, q):
                break
        else:
            filtered_detections.append(r)
    image = draw_rects_cv(image, filtered_detections)  
 
    return image


def detect_people_cascade(image):
    image = rgb_pre_processing(image)
    rects = detect_cascade(image, uppercascade)
    
    image = draw_rects_cv(image, rects,color=(0, 255, 0))  
    return image

def detectopencv_face(image):
    #frame = image.copy()
    image = pre_processing(image)

    #start = time.time()
    rects = detect_cascadeface(image)
    #rects = detect_cascadeface_accurate(image)
    #Ttime = time.time() - start

    #frame = draw_rects_cv(frame, rects)  

    #lineString = "speed: " + str(Ttime )
    #writeToFile("detections.txt",lineString)
   
    return rects

def detectlight_face(image):
    #imagep = pre_processing(image)
   
    image = rgb_pre_processing(image)
    rectscv = detect_cascadeface(image)
    processedimg = draw_rects_cv(image, rectscv)

    height, width, channels = image.shape
    rectsdlib = detectdlib_face(image,height,width)
    processedimg = draw_rects_dlib(processedimg, rectsdlib)

    cv2.imwrite('RGBlighting_Normalization.jpg',processedimg)


def detectdlibgrey_face(grey):
    bbs = detector(grey,1)
    return bbs

def detectdlib_face(img,height,width):

    buf = np.asarray(img)
    rgbFrame = np.zeros((height, width, 3), dtype=np.uint8)
    rgbFrame[:, :, 0] = buf[:, :, 2]
    rgbFrame[:, :, 1] = buf[:, :, 1]
    rgbFrame[:, :, 2] = buf[:, :, 0]

    annotatedFrame = np.copy(buf)

    # start = time.time()
    bbs = align.getAllFaceBoundingBoxes(rgbFrame)
    # Ttime = time.time() - start
    #print("Face detection took {} seconds.".format(time.time() - start))
    # lineString = "speed: " + str(Ttime )
    # writeToFile("detections.txt",lineString)

    return bbs#, annotatedFrame


def convertImageToNumpyArray(img,height,width): #numpy array used by dlib for image operations
    buf = np.asarray(img)
    rgbFrame = np.zeros((height, width, 3), dtype=np.uint8)
    rgbFrame[:, :, 0] = buf[:, :, 2]
    rgbFrame[:, :, 1] = buf[:, :, 1]
    rgbFrame[:, :, 2] = buf[:, :, 0]

    annotatedFrame = np.copy(buf)
    return annotatedFrame


def align_face(rgbFrame,bb):
    logging.debug('\n\n////////////////////// TRYING TO ALIGN FACE //////////////////////\n\n')
    landmarks = align.findLandmarks(rgbFrame, bb)
    alignedFace = align.align(args.imgDim, rgbFrame, bb,landmarks=landmarks,landmarkIndices=openface.AlignDlib.OUTER_EYES_AND_NOSE)                                                     
    if alignedFace is None:  
        print("//////////////////////  FACE COULD NOT BE ALIGNED  //////////////////////////")
        return alignedFace

    print("\n//////////////////////  FACE ALIGNED  ////////////////////// \n")
    return alignedFace

def face_recognition(alignedFace):

    with neuralNet_lock:
        logging.debug('\n\n////////////////////// TRYING TO RECOGNISE FACE //////////////////////\n\n')
        persondict = recognize_face("generated-embeddings/classifier.pkl",alignedFace, net)
        logging.debug('\n\n////////////////////// HELLO THERE RECOGNITION //////////////////////\n\n')

    if persondict is None:
        print("\n//////////////////////  FACE COULD NOT BE RECOGNIZED  //////////////////////////\n")
        return persondict
    else:
        print("\n//////////////////////  FACE RECOGNIZED  ////////////////////// \n")
        return persondict

def recognize_face(classifierModel,img,net):

    with open(classifierModel, 'r') as f:
        (le, clf) = pickle.load(f) # loads labels and classifier SVM or GMM

    if getRep(img,net) is None:  
        return None
    print("\n//////////////////////  GETTING REPRESENTATION  ////////////////////// \n")
    rep = getRep(img,net).reshape(1, -1) # gets embedding representation of image
    start = time.time()
    predictions = clf.predict_proba(rep).ravel() #Computes probabilities of possible outcomes for samples in classifier(clf).

    maxI = np.argmax(predictions)
    person1 = le.inverse_transform(maxI)
    confidence1 = int(math.ceil(predictions[maxI]*100))

    max2 = np.argsort(predictions)[-3:][::-1][1]
    person2 = le.inverse_transform(max2)
    confidence2 = int(math.ceil(predictions[max2]*100))

    print("Recognition took {} seconds.".format(time.time() - start))
    print("Recognized {} with {:.2f} confidence.".format(person1, confidence1))
    print("Second highest person: {} with {:.2f} confidence.".format(person1, confidence1))    

    # if isinstance(clf, GMM):
    #     dist = np.linalg.norm(rep - clf.means_[maxI])
    #     print("  + Distance from the mean: {}".format(dist) + " " + persondict)

    persondict = {'name': person1, 'confidence': confidence1}
    return persondict


def getRep(alignedFace,net):

    bgrImg = alignedFace
    if bgrImg is None:
        print("unable to load image")
        return None

    alignedFace = cv2.cvtColor(bgrImg, cv2.COLOR_BGR2RGB)
    start = time.time()
    print("\n//////////////////////  NEURAL NET FORWARD PASS  ////////////////////// \n")
    rep = net.forward(alignedFace)
    #print("Neural network forward pass took {} seconds.".format(  time.time() - start))
    return rep


def writeToFile(filename,lineString): #Used for writing testing data to file

       f = open(filename,"a") 
       f.write(lineString + "\n")    
       f.close()


