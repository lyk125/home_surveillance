
# Surveillance System Controller.
# Brandon Joffe
# 2016
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
# Code used in this project included opensource software (Openface)
# developed by Brandon Amos
# Copyright 2015-2016 Carnegie Mellon University


import time
import argparse
import cv2
import os
import pickle
from operator import itemgetter
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.lda import LDA
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from sklearn.mixture import GMM

import dlib

import atexit
from subprocess import Popen, PIPE
import os.path
import sys

import logging
import threading
import time
from datetime import datetime, timedelta

import smtplib
from email.MIMEMultipart import MIMEMultipart
from email.MIMEText import MIMEText
from email.MIMEBase import MIMEBase
from email import encoders

import requests
import json

import Camera
import FaceRecogniser
import openface
import aligndlib
import ImageProcessor

import random

import psutil

#pip install psutil
#pip install flask-uploads

# from instapush import Instapush, App #Used for push notifications
#Get paths for models
#//////////////////////////////////////////////////////////////////////////////////////////////
start = time.time()
np.set_printoptions(precision=2)



#///////////////////////////////////////////////////////////////////////////////////////////////
logging.basicConfig(level=logging.DEBUG,
                    format='(%(threadName)-10s) %(message)s',
                    )
                              
class Surveillance_System(object):
    
   def __init__(self):

        self.recogniser = FaceRecogniser.Face_Recogniser()

        self.training = True
        self.trainingEvent = threading.Event()
        self.trainingEvent.set()

        self.drawing = False 

        self.max_fps = 5

        self.alarmState = 'Disarmed' #disarmed, armed, triggered
        self.alarmTriggerd = False
        self.alerts = []
        self.cameras = []

        self.cameras_lock = threading.Lock()
        self.tracker_lock = threading.Lock()
        self.peopleDetected = {} #id #person object

        self.peopleDB = []

        self.camera_threads = []
        self.camera_facedetection_threads = []
        self.people_processing_threads = []
        self.svm = None

        self.video_frame1 = None
        self.video_frame2 = None
        self.video_frame3 = None

        self.fileDir = os.path.dirname(os.path.realpath(__file__))
        self.luaDir = os.path.join(self.fileDir, '..', 'batch-represent')
        self.modelDir = os.path.join(self.fileDir, '..', 'models')
        self.dlibModelDir = os.path.join(self.modelDir, 'dlib')
        self.openfaceModelDir = os.path.join(self.modelDir, 'openface')

        parser = argparse.ArgumentParser()
        parser.add_argument('--dlibFacePredictor', type=str, help="Path to dlib's face predictor.", default=os.path.join(self.dlibModelDir, "shape_predictor_68_face_landmarks.dat"))                  
        parser.add_argument('--networkModel', type=str, help="Path to Torch network model.", default=os.path.join(self.openfaceModelDir, 'nn4.small2.v1.t7'))                   
        parser.add_argument('--imgDim', type=int, help="Default image dimension.", default=96)                    
        parser.add_argument('--cuda', action='store_true')
        parser.add_argument('--unknown', type=bool, default=False, help='Try to predict unknown people')
                            
        self.args = parser.parse_args()
        self.align = openface.AlignDlib(self.args.dlibFacePredictor)
        self.net = openface.TorchNeuralNet(self.args.networkModel, imgDim=self.args.imgDim,  cuda=self.args.cuda) 

        self.confidenceThreshold = 50  #max 80
        #////////////////////////////////////////////////////Initialization////////////////////////////////////////////////////

        #self.change_alarmState()
        #self.trigger_alarm()
        
        #self.trainClassifier()  # add faces to DB and train classifier

        #default IP cam
        #self.cameras.append(Camera.VideoCamera("rtsp://admin:12345@192.168.1.64/Streaming/Channels/2"))
        #self.cameras.append(Camera.VideoCamera("rtsp://admin:12345@192.168.1.64/Streaming/Channels/2"))
        #self.cameras.append(Camera.VideoCamera("rtsp://admin:12345@192.168.1.64/Streaming/Channels/2"))
        #self.cameras.append(Camera.VideoCamera("http://192.168.1.48/video.mjpg"))
        #self.cameras.append(Camera.VideoCamera("http://192.168.1.48/video.mjpg"))
        #self.cameras.append(Camera.VideoCamera("http://192.168.1.48/video.mjpg"))
        #self.cameras.append(Camera.VideoCamera("http://192.168.1.37/video.mjpg"))
        #self.cameras.append(Camera.VideoCamera("http://192.168.1.33/video.mjpg","detect_recognise_track",False))
        #self.cameras.append(Camera.VideoCamera("debugging/iphone_distance1080pHD.m4v"))
        #self.cameras.append(Camera.VideoCamera("debugging/Test.mov"))
        #self.cameras.append(Camera.VideoCamera("debugging/Test.mov"))
        # self.cameras.append(Camera.VideoCamera("debugging/Test.mov","detect_recognise_track",False))
        # self.cameras.append(Camera.VideoCamera("debugging/Test.mov","detect_recognise_track",False))
        #self.cameras.append(Camera.VideoCamera("debugging/rotationD.m4v"))
        # self.cameras.append(Camera.VideoCamera("debugging/singleTest.m4v","detect_recognise_track",False))
        # self.cameras.append(Camera.VideoCamera("debugging/peopleTest.m4v","detect_recognise_track",False))
        # self.cameras.append(Camera.VideoCamera("debugging/singleTest.m4v","detect_recognise_track",False))
        # self.cameras.append(Camera.VideoCamera("debugging/peopleTest.m4v","detect_recognise_track",False))
        # self.cameras.append(Camera.VideoCamera("debugging/peopleTest.m4v","detect_recognise_track",False))
        #self.cameras.append(Camera.VideoCamera("debugging/example_01.mp4"))
        #self.cameras.append(Camera.VideoCamera("debugging/record.avi","detect_recognise_track",False))
        


        #self.change_alarmState()
        #self.trigger_alarm()
        self.getFaceDatabaseNames()
        #self.trainClassifier()  # add faces to DB and train classifier
        
        #processing frame threads- for detecting motion and face detection
        self.proccesing_lock = threading.Lock()

        for i, cam in enumerate(self.cameras):       
          thread = threading.Thread(name='frame_process_thread_' + str(i),target=self.process_frame,args=(cam,))
          thread.daemon = False
          self.camera_threads.append(thread)
          thread.start()

        #Thread for alert processing  
        self.alerts_lock = threading.Lock()
        thread = threading.Thread(name='alerts_process_thread_',target=self.alert_engine,args=())
        thread.daemon = False
        thread.start()


   def add_camera(self, camera):
        self.cameras.append(camera)
        thread = threading.Thread(name='frame_process_thread_' + str(len(self.cameras)),target=self.process_frame,args=(self.cameras[-1],))
        thread.daemon = False
        self.camera_threads.append(thread)
        thread.start()


   def process_frame(self,camera):
        logging.debug('Processing Frames')
        state = 1
        frame_count = 0;
        fps_count = 0
        fps_start = time.time()
        start = time.time()
        stop = False
        skip = 0
        # img = cv2.imread('debugging/lighting.jpg')
        # ImageProcessor.detectlight_face(img)
        # motion, detect_recognise, motion_detect_recognise, segment_detect_recognise, detect_recognise_track
        while True:  

             
                
             frame = camera.read_frame()
             if frame == None or np.array_equal(frame, camera.temp_frame):
                continue
             frame = ImageProcessor.resize(frame)
             height, width, channels = frame.shape

             if fps_count == 5:
                camera.processing_fps = 5/(time.time() - fps_start)
                fps_start = time.time()
                fps_count = 0


             # if camera.processing_fps != 0:
             #    time.sleep((1 - (self.max_fps/camera.processing_fps))/self.max_fps)


             fps_count += 1 
             camera.temp_frame = frame
              ##################################################################################################################################################
              #<###########################################################> MOTION DETECTION <################################################################>
              ##################################################################################################################################################

             if camera.cameraFunction == "detect_motion":

                   camera.motion, mframe = ImageProcessor.motion_detector(camera,frame, get_rects = False)   #camera.motion,

                   camera.processing_frame = mframe
                   if camera.motion == True:
                        logging.debug('\n\n////////////////////// MOTION DETECTED //////////////////////\n\n')
                        
                   else:
                        logging.debug('\n\n////////////////////// NO MOTION DETECTED //////////////////////\n\n')
                   continue

              ##################################################################################################################################################
              #<#####################################################> FACE DETECTION AND RECOGNTIION <#########################################################>
              ##################################################################################################################################################

             elif camera.cameraFunction == "detect_recognise":

                    training_blocker = self.trainingEvent.wait()  

                    frame = cv2.flip(frame, 1)
                    camera.faceBoxes = ImageProcessor.detect_faces(frame, camera.dlibDetection)
                    #camera.faceBoxes = ImageProcessor.detectdlib_face(frame,height,width)
                    if self.drawing == True:
                         frame = cv2.flip(frame, 1)
                         camera.faceBoxes = ImageProcessor.detect_faces(frame, camera.dlibDetection)
                         frame = ImageProcessor.draw_boxes(frame, camera.faceBoxes, camera.dlibDetection)
                 
                
                    #frame = ImageProcessor.draw_rects_dlib(frame, camera.faceBoxes)

                    camera.processing_frame = frame


                   
                    
                    logging.debug('\n\n//////////////////////  FACES DETECTED: '+ str(len(camera.faceBoxes)) +' //////////////////////\n\n')
                    # frame = cv2.flip(frame, 1)
                    for face_bb in camera.faceBoxes: 
                        
                        # used to reduce false positives from opencv haar cascade detector
                        if camera.dlibDetection == False:
                              x, y, w, h = face_bb
                              face_bb = dlib.rectangle(long(x), long(y), long(x+w), long(y+h))
                              faceimg = ImageProcessor.crop(frame, face_bb, dlibRect = True)
                              if len(ImageProcessor.detect_cascadeface_accurate(faceimg)) == 0:
                                    continue

                        # alignedFace = ImageProcessor.align_face(frame,face_bb)
                        # predictions = ImageProcessor.face_recognition(alignedFace)
                        predictions, alignedFace = self.recogniser.make_prediction(frame,face_bb)

                        with camera.people_dict_lock:
                          if camera.people.has_key(predictions['name']):
                              if camera.people[predictions['name']].confidence < predictions['confidence']:
                                  camera.people[predictions['name']].confidence = predictions['confidence']

                                  if camera.people[predictions['name']].confidence > self.confidenceThreshold:
                                     camera.people[predictions['name']].identity = predictions['name']
                                     if camera.unknownDetections > 0:
                                        camera.unknownDetections -= 1

                                  camera.people[predictions['name']].set_thumbnail(alignedFace) 
                                  camera.people[predictions['name']].add_to_thumbnails(alignedFace)  
                                  camera.people[predictions['name']].set_time()
                          else: 
                              if predictions['confidence'] > self.confidenceThreshold:
                                  camera.people[predictions['name']] = Person(predictions['confidence'], alignedFace, predictions['name'])
                              else: 
                                  camera.unknownDetections +=1
                                  camera.people[predictions['name']] = Person(predictions['confidence'], alignedFace, "unknown")

                   
                    camera.processing_frame = frame


              ##################################################################################################################################################
              #<#####################################> MOTION DETECTION EVENT FOLLOWED BY FACE DETECTION AND RECOGNITION <#####################################>
              ##################################################################################################################################################

             elif camera.cameraFunction == "motion_detect_recognise":

                 training_blocker = self.trainingEvent.wait()  

                 if state == 1: # if no faces have been found or there has been no movement

                     camera.motion, mframe = ImageProcessor.motion_detector(camera,frame, get_rects = False)   #camera.motion,
          
                     if camera.motion == True:
                        logging.debug('\n\n////////////////////// MOTION DETECTED //////////////////////\n\n')
                        state = 2
                        camera.processing_frame = mframe
                     else:
                        logging.debug('\n\n////////////////////// NO MOTION DETECTED //////////////////////\n\n')
                     continue

                 elif state == 2: # if motion has been detected

                    if frame_count == 0:
                        start = time.time()
                        frame_count += 1

                    frame = cv2.flip(frame, 1)
                    camera.faceBoxes = ImageProcessor.detect_faces(frame, camera.dlibDetection)
                    if self.drawing == True:
                        frame = ImageProcessor.draw_boxes(frame, camera.faceBoxes, camera.dlibDetection)
                 

                    camera.processing_frame = frame

                    if len(camera.faceBoxes) == 0:
                        if (time.time() - start) > 30.0:
                            logging.debug('\n\n//////////////////////  No faces found for ' + str(time.time() - start) + ' seconds - Going back to Motion Detection Mode\n\n')
                            state = 1
                            frame_count = 0;
                            #camera.processing_frame = frame

                    else:
                      
                        
                        logging.debug('\n\n//////////////////////  FACES DETECTED: '+ str(len(camera.faceBoxes)) +' //////////////////////\n\n')
                        # frame = cv2.flip(frame, 1)
                        for face_bb in camera.faceBoxes: 
                      
                            if camera.dlibDetection == False:
                                  x, y, w, h = face_bb
                                  face_bb = dlib.rectangle(long(x), long(y), long(x+w), long(y+h))
                                  faceimg = ImageProcessor.crop(frame, face_bb, dlibRect = True)
                                  if len(ImageProcessor.detect_cascadeface_accurate(faceimg)) == 0:
                                        continue

                            # alignedFace = ImageProcessor.align_face(frame,face_bb)
                            # predictions = ImageProcessor.face_recognition(alignedFace)

                            predictions, alignedFace = self.recogniser.make_prediction(frame,face_bb)

                            with camera.people_dict_lock:
                              if camera.people.has_key(predictions['name']):
                                  if camera.people[predictions['name']].confidence < predictions['confidence']:
                                      camera.people[predictions['name']].confidence = predictions['confidence']

                                      if camera.people[predictions['name']].confidence > self.confidenceThreshold:
                                         camera.people[predictions['name']].identity = predictions['name']
                                         if camera.unknownDetections > 0:
                                            camera.unknownDetections -= 1

                                      camera.people[predictions['name']].set_thumbnail(alignedFace)  
                                      camera.people[predictions['name']].add_to_thumbnails(alignedFace) 
                                      camera.people[predictions['name']].set_time()
                              else: 
                                  if predictions['confidence'] > self.confidenceThreshold:
                                      camera.people[predictions['name']] = Person(predictions['confidence'], alignedFace, predictions['name'])
                                  else: 
                                      camera.unknownDetections +=1
                                      camera.people[predictions['name']] = Person(predictions['confidence'], alignedFace, "unknown")

                        start = time.time() #used to go back to motion detection state of 20s of not finding a face
                        camera.processing_frame = frame

              ###################################################################################################################################################################
              #<#####################################>  MOTION DETECTION OBJECT SEGMENTAION FOLLOWED BY FACE DETECTION AND RECOGNITION <#####################################>
              ####################################################################################################################################################################
      
             elif camera.cameraFunction == "segment_detect_recognise":
                    # if motion has been detected use segmented region to detect a face
                    training_blocker = self.trainingEvent.wait() 
                    camera.motion, peopleRects  = ImageProcessor.motion_detector(camera,frame, get_rects = True)   #camera.motion,
                    #camera.processing_frame = frame
                    if camera.motion == False:
                       camera.processing_frame = frame
                       logging.debug('\n\n//////////////////////-- NO MOTION DETECTED --//////////////////////\n\n')
                       continue

                    logging.debug('\n\n////////////////////// MOTION DETECTED //////////////////////\n\n')
                    if self.drawing == True:
                        frame = ImageProcessor.draw_boxes(frame, peopleRects, False)

                    for x, y, w, h in peopleRects:
                      
                        logging.debug('\n\n////////////////////// Proccessing People Segmented Areas //////////////////////\n\n')
                        bb = dlib.rectangle(long(x), long(y), long(x+w), long(y+h)) 
                        personimg = ImageProcessor.crop(frame, bb, dlibRect = True)
                       
                        personimg = cv2.flip(personimg, 1)
                        camera.faceBoxes = ImageProcessor.detect_faces(personimg, camera.dlibDetection)
                        if self.drawing == True:
                            camera.processing_frame = ImageProcessor.draw_boxes(frame, peopleRects, False)

                        for face_bb in camera.faceBoxes: 

                              if camera.dlibDetection == False:
                                    x, y, w, h = face_bb
                                    face_bb = dlib.rectangle(long(x), long(y), long(x+w), long(y+h))
                                    faceimg = ImageProcessor.crop(personimg, face_bb, dlibRect = True)
                                    if len(ImageProcessor.detect_cascadeface_accurate(faceimg)) == 0:
                                          continue

                              logging.debug('\n\n////////////////////// Proccessing Detected faces //////////////////////\n\n')
                            
                              # alignedFace = ImageProcessor.align_face(personimg,face_bb)
                              # predictions = ImageProcessor.face_recognition(alignedFace)

                              predictions, alignedFace = self.recogniser.make_prediction(personimg,face_bb)

                              with camera.people_dict_lock:
                                if camera.people.has_key(predictions['name']):
                                    if camera.people[predictions['name']].confidence < predictions['confidence']:
                                        camera.people[predictions['name']].confidence = predictions['confidence']
                                        camera.people[predictions['name']].set_thumbnail(alignedFace)  
                                        camera.people[predictions['name']].add_to_thumbnails(alignedFace) 
                                        camera.people[predictions['name']].set_time()
                                else: 
                                    if predictions['confidence'] > self.confidenceThreshold:
                                        camera.people[predictions['name']] = Person(predictions['confidence'], alignedFace, predictions['name'])
                                    else: 
                                        camera.unknownDetections +=1
                                        camera.people[predictions['name']] = Person(predictions['confidence'], alignedFace, "unknown")
              
              ############################################################################################################################################################################
              #<#####################################>  MOTION DETECTION OBJECT SEGMENTAION FOLLOWED BY FACE DETECTION, RECOGNITION AND TRACKING <#####################################>
              #############################################################################################################################################################################

             elif camera.cameraFunction == "detect_recognise_track":

                training_blocker = self.trainingEvent.wait()  

                logging.debug('\n\n////////////////////// detect_recognise_track 1 //////////////////////\n\n')
                peopleFound = False
                camera.motion, peopleRects  = ImageProcessor.motion_detector(camera,frame, get_rects = True)   #camera.motion,
                logging.debug('\n\n////////////////////// detect_recognise_track  2 //////////////////////\n\n')
                if self.drawing == True:
                    camera.processing_frame = ImageProcessor.draw_boxes(frame, peopleRects, False)
                #frame = cv2.flip(frame, 1)
                #camera.processing_frame = frame
                if camera.motion == False:
                   camera.processing_frame = frame
                   logging.debug('\n\n////////////////////// NO MOTION DETECTED //////////////////////\n\n')
                   continue

                logging.debug('\n\n////////////////////// MOTION DETECTED //////////////////////\n\n')

                 #<#####################################> LOOKING FOR FACES IN BOUNDING BOXES <#####################################>
                for x, y, w, h in peopleRects:
                    peopleFound = True
                    #frame = cv2.flip(frame, 1)
                    person_bb = dlib.rectangle(long(x), long(y), long(x+w), long(y+h)) 
                    personimg = ImageProcessor.crop(frame, person_bb, dlibRect = True)

                    #camera.faceBoxes = ImageProcessor.detect_cascadeface(personimg)
                    #personimg = cv2.flip(personimg, 1)
                    #camera.faceBoxes = ImageProcessor.detectdlib_face_rgb(personimg)

                    personimg = cv2.flip(personimg, 1)
                    camera.faceBoxes = ImageProcessor.detect_faces(personimg, camera.dlibDetection)
                   
                    logging.debug('\n\n//////////////////////  FACES DETECTED: '+ str(len(camera.faceBoxes)) +' //////////////////////\n\n')

                  

                    tracked = False
                     #<#####################################> ITERATING THROUGH EACH TRACKER <#####################################>
                    for i in xrange(len(camera.trackers) - 1, -1, -1): # starts with the most recent tracker
                        
                        if camera.trackers[i].overlap(person_bb):
                           print "============================> Updating Tracker <============================"
                           camera.trackers[i].updateTracker(person_bb)
                                                    
                           for face_bb in camera.faceBoxes: 

                                if camera.dlibDetection == False:
                                    x, y, w, h = face_bb
                                    face_bb = dlib.rectangle(long(x), long(y), long(x+w), long(y+h))
                                    faceimg = ImageProcessor.crop(personimg, face_bb, dlibRect = True)
                                    if len(ImageProcessor.detect_cascadeface_accurate(faceimg)) == 0:
                                          continue
          
                                # alignedFace = ImageProcessor.align_face(personimg,face_bb)
                                # predictions = ImageProcessor.face_recognition(alignedFace)

                                predictions, alignedFace =  self.recogniser.make_prediction(personimg,face_bb)

                                #<#####################################> ONLY ONE FACE DETECTED <#####################################>
                                if len(camera.faceBoxes) == 1:
                                    # if not the same person check to see if tracked person is unknown and update or change tracker accordingly
                                    if camera.trackers[i].person.identity != predictions['name']: 
                                        if camera.trackers[i].person.identity == "unknown":
                                            print "hellothere1 " + predictions['name'] + " " + str(predictions['confidence'])
                                            if camera.trackers[i].person.confidence < predictions['confidence']:
                                              camera.trackers[i].person.confidence = predictions['confidence']
                                              if camera.trackers[i].person.confidence > self.confidenceThreshold:
                                                  camera.trackers[i].person.identity = predictions['name']
                                        else:
                                             print "hellothere12 " + predictions['name'] + " " + str(predictions['confidence'])
                                             if self.confidenceThreshold < predictions['confidence']:
                                                   strID = "person" +  datetime.now().strftime("%Y%m%d%H%M%S") + str(num)
                                                   if predictions['confidence'] > self.confidenceThreshold:
                                                              person = Person(predictions['confidence'], alignedFace, predictions['name'])
                                                   else:   
                                                              person = Person(predictions['confidence'], alignedFace, "unknown")
                                                   # with camera.people_dict_lock:
                                                   #            camera.people[strID] = person
                                                   print "============================> Changing Tracker X <============================"
                                                   camera.trackers[i] = Tracker(frame, person_bb, person,strID)

                                    # if it is the same person update confidence if it is higher and change prediction from unknown to identified person
                                    # if the new detected face has a lower confidence and can be classified as unknown, when the person being tracked isn't unknown - change tracker
                                    else:  
                                        if camera.trackers[i].person.confidence < predictions['confidence']:
                                            camera.trackers[i].person.confidence = predictions['confidence']
                                            if camera.trackers[i].person.confidence > self.confidenceThreshold:
                                                camera.trackers[i].person.identity = predictions['name']
                                        else:
                                            print "hellothere123 " + predictions['name'] + " " + str(predictions['confidence'])
                                            if self.confidenceThreshold*0.7 > predictions['confidence']:
                                                 strID = "person" +  datetime.now().strftime("%Y%m%d%H%M%S") + str(num)
                                                 if predictions['confidence'] > self.confidenceThreshold:
                                                        person = Person(predictions['confidence'], alignedFace, predictions['name'])
                                                 else:   
                                                        person = Person(predictions['confidence'], alignedFace, "unknown")
                                                 # with camera.people_dict_lock:
                                                 #        camera.people[strID] = person
                                                 print "============================> Changing Tracker Y <============================"
                                                 camera.trackers[i] = Tracker(frame, person_bb, person,strID)
                                      
                                 #<#####################################> MORE THAN ONE FACE DETECTED <#####################################>
                                else:
                                    print "============================> More Than One Face Detected <============================"
                                    # if tracker is already tracking the identified face make an update 
                                    if camera.trackers[i].person.identity == predictions['name']:

                                        if camera.trackers[i].person.confidence < predictions['confidence']:
                                            camera.trackers[i].person.confidence = predictions['confidence']
                                            if camera.trackers[i].person.confidence > self.confidenceThreshold:
                                                camera.trackers[i].person.identity = predictions['name']
                                    else:
                                        # if tracker isn't tracking this face check the next tracker
                                        break
                                   
                                camera.trackers[i].person.set_thumbnail(alignedFace)  
                                camera.trackers[i].person.add_to_thumbnails(alignedFace)
                                camera.trackers[i].person.set_time()
                                camera.trackers[i].resetFacePinger()
                                with camera.people_dict_lock:
                                        camera.people[camera.trackers[i].id] = camera.trackers[i].person
                           camera.trackers[i].resetPinger()
                           tracked = True
                           break

                     #<#####################################> IF THE PERSON BOUNDING BOX IS NOT BEING TRACKED <#####################################>
                    if not tracked:

                        #personimg = ImageProcessor.crop(frame, person_bb, dlibRect = True)
                        #camera.faceBoxes = ImageProcessor.detect_cascadeface(personimg)
                        #<#####################################> LOOK FOR FACES IN BOUNDING BOX <#####################################>
                        camera.faceBoxes = ImageProcessor.detect_faces(personimg, camera.dlibDetection)
                        for face_bb in camera.faceBoxes: 
                        #for x, y, w, h in camera.faceBoxes: 
                            #face_bb = dlib.rectangle(long(x), long(y), long(x+w), long(y+h))
                            num = random.randrange(1, 1000, 1)    #unique ID
                            strID = "person" +  datetime.now().strftime("%Y%m%d%H%M%S") + str(num)

                            if camera.dlibDetection == False:
                                  x, y, w, h = face_bb
                                  face_bb = dlib.rectangle(long(x), long(y), long(x+w), long(y+h))
                                  faceimg = ImageProcessor.crop(personimg, face_bb, dlibRect = True)
                                  if len(ImageProcessor.detect_cascadeface_accurate(faceimg)) == 0:
                                        continue
                      
                            # alignedFace = ImageProcessor.align_face(personimg,face_bb)
                            # predictions = ImageProcessor.face_recognition(alignedFace)
                            predictions, alignedFace =  self.recogniser.make_prediction(personimg,face_bb)

                            if predictions['confidence'] > self.confidenceThreshold:
                                  person = Person(predictions['confidence'], alignedFace, predictions['name'])
                            else:   
                                  person = Person(predictions['confidence'], alignedFace, "unknown")
                                  camera.unknownDetections +=1
                            #add person to detected people      
                            with camera.people_dict_lock:
                                  camera.people[strID] = person
                            print "============================> New Tracker <============================"
                             #<#####################################> CREATE NEW TRACKER FOR EVERY PERSON/FACE DETECTED <#####################################>
                            camera.trackers.append(Tracker(frame, person_bb, person,strID))

                print len(camera.trackers)
                for i in xrange(len(camera.trackers) - 1, -1, -1): # starts with the most recently initiated tracker
                    if self.drawing == True:
                          bl = (camera.trackers[i].bb.left(), camera.trackers[i].bb.bottom()) # (x, y)
                          tr = (camera.trackers[i].bb.right(), camera.trackers[i].bb.top()) # (x+w,y+h)
                          cv2.rectangle(frame, bl, tr, color=(0, 255, 255), thickness=2)
                          cv2.putText(frame,  camera.trackers[i].person.identity + " " + str(camera.trackers[i].person.confidence)+ "%", (camera.trackers[i].bb.left()+10, camera.trackers[i].bb.bottom() - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.3,
                                    color=(0, 255, 255), thickness=1)
                    camera.processing_frame = frame
                    t = camera.trackers[i]
                    camera.trackers[i].ping()
                    


                    if camera.trackers[i].pings > 20: 
                        with self.tracker_lock:
                          del camera.trackers[i]
                        continue

                    if camera.trackers[i].facepings > 70:
                        with self.tracker_lock:
                              del camera.trackers[i]
                        continue
                        camera.trackers[i].faceping()

   #<#####################################> DLIBS CORRELATION TRACKER - didnt work too well... <#####################################>
                # for x, y, w, h in peopleRects:
                #     peopleFound = True
                #     #frame = cv2.flip(frame, 1)
                #     person_bb = dlib.rectangle(long(x), long(y), long(x+w), long(y+h)) 
                #     personimg = ImageProcessor.crop(frame, person_bb, dlibRect = True)
                #     #camera.faceBoxes = ImageProcessor.detect_cascadeface(personimg)
                #     personimg = cv2.flip(personimg, 1)

                #     training_blocker = self.trainingEvent.wait()  

                #     tracked = False
                #     for i in xrange(len(camera.trackers) - 1, -1, -1): # starts with the most recent tracker
                #         camera.trackers[i].updateTracker(frame)
                #         if camera.trackers[i].overlap(person_bb):
                #            print "============================> Updating Tracker <============================"
                          
                #            camera.faceBoxes = ImageProcessor.detectdlib_face_rgb(personimg)
                #            for face_bb in camera.faceBoxes: 
                #            #for x, y, w, h in camera.faceBoxes: 
                #                 logging.debug('\n\n//////////////////////  FACES DETECTED: '+ str(len(camera.faceBoxes)) +' //////////////////////\n\n')
                #                 #face_bb = dlib.rectangle(long(x), long(y), long(x+w), long(y+h))
                #                 alignedFace = ImageProcessor.align_face(personimg,face_bb)
                #                 predictions = ImageProcessor.face_recognition(alignedFace)
                          
                #                 if camera.trackers[i].person.confidence < predictions['confidence']:
                #                     camera.trackers[i].person.confidence = predictions['confidence']
                #                 if camera.trackers[i].person.confidence > self.confidenceThreshold:
                #                     camera.trackers[i].person.identity = predictions['name']
                                   
                #                 camera.trackers[i].person.set_thumbnail(alignedFace)  
                #                 camera.trackers[i].person.set_time()
                #            if len(camera.faceBoxes) > 0:
                #                with camera.people_dict_lock:
                #                    camera.people[camera.trackers[i].id] = camera.trackers[i].person
                #                    camera.trackers[i].resetFacePinger()
                #            camera.trackers[i].resetPinger()
                #            tracked = True
                #            break

                #     if not tracked:
                #         num = random.randrange(1, 1000, 1)    #unique ID
                #         strID = "person" +  datetime.now().strftime("%Y%m%d%H%M%S") + str(num)

                #         personimg = ImageProcessor.crop(frame, person_bb, dlibRect = True)
                #         #camera.faceBoxes = ImageProcessor.detect_cascadeface(personimg)
                #         personimg = cv2.flip(personimg, 1)
                #         camera.faceBoxes = ImageProcessor.detectdlib_face_rgb(personimg)
                #         for face_bb in camera.faceBoxes: 
                #         #for x, y, w, h in camera.faceBoxes: 
                #             #face_bb = dlib.rectangle(long(x), long(y), long(x+w), long(y+h))
                #             alignedFace = ImageProcessor.align_face(personimg,face_bb)
                #             predictions = ImageProcessor.face_recognition(alignedFace)
                #             if predictions['confidence'] > self.confidenceThreshold:
                #                   person = Person(predictions['confidence'], alignedFace, predictions['name'])
                #             else:   
                #                   person = Person(predictions['confidence'], alignedFace, "unknown")
                #             with camera.people_dict_lock:
                #                   camera.people[strID] = person
                #             print "============================> New Tracker <============================"
                #             camera.trackers.append(Tracker(frame, person_bb, person,strID))
                #             break
                      
                # # if peopleFound == False:
                # #   for i in xrange(len(camera.trackers) - 1, -1, -1): # starts with the most recent tracker
                # #           camera.trackers[i].updateTracker(frame)

                # for i in xrange(len(camera.trackers) - 1, -1, -1): # starts with the most recently initiated tracker
                #     bl = (camera.trackers[i].bb.left(), camera.trackers[i].bb.bottom()) # (x, y)
                #     tr = (camera.trackers[i].bb.right(), camera.trackers[i].bb.top()) # (x+w,y+h)
                #     cv2.rectangle(frame, bl, tr, color=(0, 255, 255), thickness=2)
                #     camera.processing_frame = frame
                #     t = camera.trackers[i]
                #     camera.trackers[i].ping()
                    
                #     # if camera.trackers[i].person.face == None: #if a face has not been detected using tracker for 10 frames delete tracker
                #     #     if camera.trackers[i].facepings > 40:
                #     #         with self.tracker_lock:
                #     #           del camera.trackers[i]
                #     #         continue
                #     #     camera.trackers[i].faceping()

                #     if camera.trackers[i].pings > 20: 
                #         with self.tracker_lock:
                #           del camera.trackers[i]
                #         continue

                #     for j in range(i): # if any of the trackers overlap eachother with more than 40% delete 
                #         if t.facetracker.get_position().intersect(camera.trackers[j].facetracker.get_position()).area() / t.facetracker.get_position().area() > 0.3:
                #               with self.tracker_lock:
                #                 del camera.trackers[i]
                #               continue
                   
                   
                    #camera.rgbFrame = ImageProcessor.convertImageToNumpyArray(frame,height,width) # conversion required by dlib methods                  
                    #camera.processing_frame = ImageProcessor.draw_rects_dlib(frame, camera.faceBoxes)
            
   def alert_engine(self):     #check alarm state -> check camera -> check event -> either look for motion or look for detected faces -> take action
        logging.debug('Alert engine starting')
        while True:
           with self.alerts_lock:
              for alert in self.alerts:
                logging.debug('\nchecking alert\n')
                if alert.action_taken == False: #if action hasn't been taken for event 
                    if alert.alarmState != 'All':  #check states
                        if  alert.alarmState == self.alarmState: 
                            logging.debug('checking alarm state')
                            alert.event_occurred = self.check_camera_events(alert)
                        else:
                          continue # alarm not in correct state check next alert
                    else:
                        alert.event_occurred = self.check_camera_events(alert)
                else:
                    if (time.time() - alert.eventTime) > 300: # reinitialize event 5 min after event accured
                        print "reinitiallising alert: " + alert.id
                        alert.reinitialise()
                    continue 

           time.sleep(2) #put this thread to sleep - let websocket update alerts if need be (delete or add)
  
   def check_camera_events(self,alert):   

        if alert.camera != 'All':  #check cameras               
            if alert.event == 'Recognition': #Check events
                print  "checkingalertconf "+ str(alert.confidence) + " : " + alert.person
                for person in self.cameras[int(alert.camera)].people.values():
                    print "checkingalertconf "+ str(alert.confidence )+ " : " + alert.person + " : " + person.identity
                    if alert.person == person.identity: # has person been detected
                       
                        if alert.person == "unknown" and (100 - person.confidence) >= alert.confidence:
                            cv2.imwrite("notification/image.png", self.cameras[int(alert.camera)].processing_frame)#
                            self.take_action(alert)
                            return True
                        elif person.confidence >= alert.confidence:
                            cv2.imwrite("notification/image.png", self.cameras[int(alert.camera)].processing_frame)#
                            self.take_action(alert)
                            return True
         
                return False # person has not been detected check next alert       

            else:
                if self.cameras[int(alert.camera)].motion == True: # has motion been detected
                       cv2.imwrite("notification/image.png", self.cameras[int(alert.camera)].processing_frame)#
                       self.take_action(alert)
                       return True
                else:
                  return False # motion was not detected check next alert
        else:
            if alert.event == 'Recognition': #Check events
                with  self.cameras_lock:
                    for camera in self.cameras: # look through all cameras
                        for person in camera.people.values():
                            if alert.person == person.identity: # has person been detected
                                if alert.person == "unknown" and (100 - person.confidence) >= alert.confidence:
                                    cv2.imwrite("notification/image.png", self.cameras[int(alert.camera)].processing_frame)#
                                    self.take_action(alert)
                                    return True
                                elif person.confidence >= alert.confidence:
                                    cv2.imwrite("notification/image.png", self.cameras[int(alert.camera)].processing_frame)#
                                    self.take_action(alert)
                                    return True
               
                return False # person has not been detected check next alert   

            else:
                with  self.cameras_lock:
                    for camera in self.cameras: # look through all cameras
                        if camera.motion == True: # has motion been detected
                            cv2.imwrite("notification/image.png", camera.processing_frame)#
                            self.take_action(alert)
                            return True

                return False # motion was not detected check next camera

   def take_action(self,alert): 
        print "Taking action: ======================================================="
        print alert.actions
        print "======================================================================"
        if alert.action_taken == False: #only take action if alert hasn't accured 
            alert.eventTime = time.time()  
            if alert.actions['email_alert'] == 'true':
                print "\nemail notification being sent\n"
                self.send_email_notification_alert(alert)
            if alert.actions['trigger_alarm'] == 'true':
                print "\ntriggering alarm\n"
                self.trigger_alarm()
            # if alert.actions['notify_police'] == 'true':
            #     print "\nnotifying police\n"
                #notify police
            #if alert.actions['push_alert'] == 'true':
                #     print "\npush notification being sent\n"
                #     self.send_push_notification(alert.alertString)
            alert.action_taken = True


   # def trainClassifier(self):

   #      self.trainingEvent.clear() #event used to hault face_processing threads to ensure no threads try access .pkl file while it is being updated
   #      time.sleep(0.2) # delay so that threads can finish processing 
   #      path = self.fileDir + "/aligned-images/cache.t7" 
   #      try:
   #        os.remove(path) # remove cache from aligned images folder
   #      except:
   #        print "Tried to remove cache.t7"
   #        pass

   #      start = time.time()
   #      aligndlib.alignMain("training-images/","aligned-images/","outerEyesAndNose",self.args.dlibFacePredictor,self.args.imgDim)
   #      print("\nAligning images took {} seconds.".format(time.time() - start))
          
   #      done = False
   #      start = time.time()

   #      done = self.generate_representation()
           
   #      if done is True:
   #          print("Representation Generation (Classification Model) took {} seconds.".format(time.time() - start))
   #          start = time.time()
   #          #Train Model
   #          self.train("generated-embeddings/","LinearSvm",-1)
   #          print("Training took {} seconds.".format(time.time() - start))
   #      else:
   #          print("Generate representation did not return True")

   #      reloaded = self.recogniser.reloadClassifier()

   #      self.trainingEvent.set() #threads can continue processing

   #      return True
      
   # def generate_representation(self):
   #      #2 Generate Representation 
   #      print "\n" + self.luaDir + "\n"
   #      self.cmd = ['/usr/bin/env', 'th', os.path.join(self.luaDir, 'main.lua'),'-outDir',  "generated-embeddings/" , '-data', "aligned-images/"]                 
   #      if self.args.cuda:
   #          self.cmd.append('-cuda')
   #      self.p = Popen(self.cmd, stdin=PIPE, stdout=PIPE, bufsize=0)
   #      result = self.p.wait()  # wait for subprocess to finish writing to files - labels.csv reps.csv

   #      def exitHandler():
   #          if self.p.poll() is None:
   #              print "======================Something went Wrong============================"
   #              self.p.kill()
   #              return False
   #      atexit.register(exitHandler) 

   #      return True

   # def train(self,workDir,classifier,ldaDim):
   #    print("Loading embeddings.")
   #    fname = "{}/labels.csv".format(workDir) #labels of faces
   #    labels = pd.read_csv(fname, header=None).as_matrix()[:, 1]
   #    labels = map(itemgetter(1),
   #                 map(os.path.split,
   #                     map(os.path.dirname, labels)))  

   #    fname = "{}/reps.csv".format(workDir) #representations of faces
   #    embeddings = pd.read_csv(fname, header=None).as_matrix() #get embeddings as a matrix from reps.csv
   #    le = LabelEncoder().fit(labels) # encodes 
   #    labelsNum = le.transform(labels)
   #    nClasses = len(le.classes_)
   #    print("Training for {} classes.".format(nClasses))

   #    if classifier == 'LinearSvm':
   #        clf = SVC(C=1, kernel='linear', probability=True)
   #    elif classifier == 'GMM':
   #        clf = GMM(n_components=nClasses)

   #    if ldaDim > 0:
   #        clf_final = clf
   #        clf = Pipeline([('lda', LDA(n_components=ldaDim)),
   #                        ('clf', clf_final)])

   #    clf.fit(embeddings, labelsNum) #link embeddings to labels

   #    fName = "{}/classifier.pkl".format(workDir)
   #    print("Saving classifier to '{}'".format(fName))
   #    with open(fName, 'w') as f:
   #        pickle.dump((le, clf), f) # creates character stream and writes to file to use for recognition

   def send_email_notification_alert(self,alert):
      # code produced in this tutorial - http://naelshiab.com/tutorial-send-email-python/
      fromaddr = "home.face.surveillance@gmail.com"
      toaddr = alert.emailAddress

      msg = MIMEMultipart()
       
      msg['From'] = fromaddr
      msg['To'] = toaddr
      msg['Subject'] = "HOME SURVEILLANCE"
       
      body = "NOTIFICATION ALERT:\n\n" +  alert.alertString + "\n\n"
       
      msg.attach(MIMEText(body, 'plain'))
       
      filename = "image.png"
      attachment = open("notification/image.png", "rb")
               
       
      part = MIMEBase('application', 'octet-stream')
      part.set_payload((attachment).read())
      encoders.encode_base64(part)
      part.add_header('Content-Disposition', "attachment; filename= %s" % filename)
       
      msg.attach(part)
       
      server = smtplib.SMTP('smtp.gmail.com', 587)
      server.starttls()
      server.login(fromaddr, "facialrecognition")
      text = msg.as_string()
      server.sendmail(fromaddr, toaddr, text)
      server.quit()

   # def send_push_notification (self,alarmMesssage): # pip install instapush 

   #    #insta = Instapush(user_token='57c5f710a4c48a6d45ee19ce')

   #    #insta.list_app()             #List all apps

   #    #insta.add_app(title='Home Surveillance') #Create a app named title

   #    app = App(appid='57c5f92aa4c48adc4dee19ce', secret='2ed5c7b8941214510a94cfe4789ddb9f')

   #    #app.list_event()             #List all event

   #    #app.add_event(event_name='FaceDetected', trackers=['message'],
   #    #              message='{message} face detected.')

   #    app.notify(event_name='FaceDetected', trackers={'message': "NOTIFICATION ALERT\n_______________________\n" +  alarmMesssage})

   def add_face(self,name,image, upload):

      if upload == False:
          path = self.fileDir + "/aligned-images/" 
      else:
          path = self.fileDir + "/training-images/"         
      num = 0
    
      if not os.path.exists(path + name):
        try:
          print "Creating New Face Dircectory: " + name
          os.makedirs(path+name)
        except OSError:
          print OSError
          return False
          pass
      else:
         num = len([nam for nam in os.listdir(path +name) if os.path.isfile(os.path.join(path+name, nam))])

      print "Writing Image To Directory: " + name
      cv2.imwrite(path+name+"/"+ name + "-"+str(num) + ".png", image)
      self.getFaceDatabaseNames()

      return True


   def getFaceDatabaseNames(self):

      path = self.fileDir + "/aligned-images/" 
      self.peopleDB = []
      for name in os.listdir(path):
        if (name == 'cache.t7' or name == '.DS_Store' or name[0:7] == 'unknown'):
          continue
        self.peopleDB.append(name)
        print name
      self.peopleDB.append('unknown')

   def change_alarmState(self):
      r = requests.post('http://192.168.1.35:5000/change_state', data={"password": "admin"})
      alarm_states = json.loads(r.text) 
    
      print alarm_states

      if alarm_states['state'] == 1:
          self.alarmState = 'Armed' 
      else:
          self.alarmState = 'Disarmed' 
       
      self.alarmTriggerd = alarm_states['triggered']

   def trigger_alarm(self):

      r = requests.post('http://192.168.1.35:5000/trigger', data={"password": "admin"})
      alarm_states = json.loads(r.text) 
    
      print alarm_states

      if alarm_states['state'] == 1:
         self.alarmState = 'Armed' 
      else:
         self.alarmState = 'Disarmed' 
       
      self.alarmTriggerd = alarm_states['triggered']
      print self.alarmTriggerd 



#\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\
class Person(object):
    person_count = 0

    def __init__(self,confidence = 0, face = None, name = "unknown"):  

        self.identity = name
        self.count = Person.person_count
        self.confidence = confidence  
        self.thumbnails = []
        self.face = face
        if face is not None:
            ret, jpeg = cv2.imencode('.jpg', face) #convert to jpg to be viewed by client
            self.thumbnail = jpeg.tostring()
        self.thumbnails.append(self.thumbnail) 
        Person.person_count += 1 
        now = datetime.now() + timedelta(hours=2)
        self.time = now.strftime("%A %d %B %Y %I:%M:%S%p")

        print self.time 

    def get_prediction(self):
        # if (self.identity == "unknown"):
            return self.identity #+ "_" + str(self.count)
        # else:
        #     return self.identity 

    def set_identity(self, identity):
        self.identity = identity

    def set_time(self):
        now = datetime.now() + timedelta(hours=2)
        self.time = now.strftime("%A %d %B %Y %I:%M:%S%p")

    def set_thumbnail(self, face):
        self.face = face
        ret, jpeg = cv2.imencode('.jpg', face) #convert to jpg to be viewed by client
        self.thumbnail = jpeg.tostring()

    def add_to_thumbnails(self, face):
        ret, jpeg = cv2.imencode('.jpg', face) #convert to jpg to be viewed by client
        self.thumbnails.append(jpeg.tostring())

class Tracker:
    tracker_count = 0

    def __init__(self, img, bb, person, id):
        self.id = id 
        self.person = person
        self.bb = bb
        self.pings = 0
        self.facepings = 0

    def resetPinger(self):
        self.pings = 0

    def resetFacePinger(self):
        self.facepings = 0

    def updateTracker(self,bb):
        self.bb  = bb 
        
    def overlap(self, bb):
        p = float(self.bb.intersect(bb).area()) / float(self.bb.area())
        return p > 0.4

    def ping(self):
        self.pings += 1

    def faceping(self):
        self.facepings += 1

# class Tracker:
#     tracker_count = 0

#     def __init__(self, img, bb, person, id):
#         self.facetracker = dlib.correlation_tracker()
#         self.facetracker.start_track(img, bb)
#         self.id = id 
#         self.person = person
#         self.bb = bb
#         self.pings = 0
#         self.facepings = 0

#     def resetPinger(self):
#         self.pings = 0

#     def resetFacePinger(self):
#         self.facepings = 0

#     def updateTracker(self,img):
#         self.facetracker.update(img)
#         box = self.facetracker.get_position() #put position in usable format
#         self.bb =  dlib.rectangle(long(box.left()), long(box.top()), long(box.right()), long(box.bottom())) 
       
#     def overlap(self, bb):
#         p = float(self.bb.intersect(bb).area()) / float(self.bb.area())
#         return p > 0.3

#     def ping(self):
#         self.pings += 1

#     def faceping(self):
#         self.facepings += 1



class Alert(object): #array of alerts   alert(camera,alarmstate(use switch statements), event(motion recognition),)

    alert_count = 1

    def __init__(self,alarmState,camera, event, person, actions, emailAddress, confidence):   
        print "\n\nalert_"+str(Alert.alert_count)+ " created\n\n"
       

        if  event == 'Motion':
            self.alertString = "Motion detected in camera " + camera 
        else:
            self.alertString = person + " was recognised in camera " + camera + " with a confidence greater than " + str(confidence)

        self.id = "alert_"+str(Alert.alert_count)
        self.event_occurred = False
        self.action_taken = False
        self.camera = camera
        self.alarmState = alarmState
        self.event = event
        self.person = person
        self.confidence = confidence
        self.actions = actions
        if emailAddress == None:
            self.emailAddress = "bjjoffe@gmail.com"
        else:
            self.emailAddress = emailAddress

        self.eventTime = 0

        Alert.alert_count += 1

    def reinitialise(self):
        self.event_occurred = False
        self.action_taken = False

    def set_custom_alertmessage(self,message):
        self.alertString = message

        



