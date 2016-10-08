# Camera Class
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
# Some of the code for this VideoCamera class was based on:
# Raspberry Pi Face Recognition Treasure Box 
# Webcam OpenCV Camera Capture Device
# Copyright 2013 Tony DiCola 


import threading
import time
import numpy as np
import cv2
import ImageProcessor
import dlib
import openface
import os
import argparse

import logging

# from flask import Flask, render_template, Response, redirect, url_for, request
import Camera
# from flask.ext.socketio import SocketIO,send, emit #Socketio depends on gevent
import SurveillanceSystem

CAPTURE_HZ = 25.0

logging.basicConfig(level=logging.DEBUG,
                    format='(%(threadName)-10s) %(message)s',
                    )


class VideoCamera(object):
    def __init__(self,camURL, cameraFunction, dlibDetection):
		print("Loading Stream From IP Camera ",camURL)

		self.processing_frame = None
		self.temp_frame = None
		self.capture_frame = None

		self.streaming_fps = 0
		self.processing_fps = 0

		self.fps_start = time.time()
		self.fps_count = 0

		self.motion = False     # used for alerts and transistion between system states i.e from motion detection to face detection
		self.people = {}		# holds detected faces as well as there condidences
		self.trackers = []
		self.unknownDetections = 0

		self.cameraFunction = cameraFunction # "detect_recognise_track"    # detect_motion, detect_recognise, motion_detect_recognise, segment_detect_recognise, detect_recognise_track

		self.dlibDetection = dlibDetection  #used to choose detection method for camera (dlib - True vs opencv - False)

		self.previous_frame = None
		self.current_frame = None
		self.next_frame = None
		self.history = 0
		self.meanframe = None	

		self.frame_counter = 0
		self.rgbFrame = None
		self.faceBoxes = None

		self.captureEvent = threading.Event()
		self.captureEvent.set()

		self.people_dict_lock = threading.Lock()

		self.frame_lock = threading.Lock()
		print camURL
	 	self.video = cv2.VideoCapture(camURL)
	 	self.url = camURL
		if not self.video.isOpened():
			self.video.open()
			
		# Start a thread to continuously capture frames.
		# Use a lock to prevent concurrent access to the camera.
		# The capture thread ensures the frames being processed are up to date and are not old
		self.capture_lock = threading.Lock()
		self.capture_thread = threading.Thread(name='video_capture_thread',target=self.get_frame)
		self.capture_thread.daemon = True
		self.capture_thread.start()

		# Neural Net Object
		fileDir = os.path.dirname(os.path.realpath(__file__))
		modelDir = os.path.join(fileDir, '..', 'models')
		dlibModelDir = os.path.join(modelDir, 'dlib')
		openfaceModelDir = os.path.join(modelDir, 'openface')
		parser = argparse.ArgumentParser()
		parser.add_argument('--networkModel', type=str, help="Path to Torch network model.",
                    default=os.path.join(openfaceModelDir, 'nn4.small2.v1.t7'))
		parser.add_argument('--imgDim', type=int,
                    help="Default image dimension.", default=96)
		parser.add_argument('--cuda', action='store_true')
		args = parser.parse_args()

                                  
    def __del__(self):
        self.video.release()
    	
    def get_frame(self):
		logging.debug('Getting Frames')
		fps_count = 0
		fps_start = time.time()
		while True:
			success, frame = self.video.read()
			self.captureEvent.clear() 
			#with self.capture_lock:   # stops other processes from reading the frame while it is being updated
				#self.capture_frame = None
			if success:		
				self.capture_frame = frame
				self.captureEvent.set() 

			fps_count += 1 

			if fps_count == 5:
				self.streaming_fps = 5/(time.time() - fps_start)
				fps_start = time.time()
				fps_count = 0

			if self.streaming_fps != 0:  # if frame rate gets too fast slow it down, if it gets too slow speed it up
				if self.streaming_fps > CAPTURE_HZ:
					time.sleep(1/CAPTURE_HZ) 
				else:
					time.sleep(self.streaming_fps/(CAPTURE_HZ*CAPTURE_HZ))
			
			
			 # time.sleep(1.0/CAPTURE_HZ) # only sleep if frame captured


    def read_jpg(self):

		#frame = None
		#with self.capture_lock:
		capture_blocker = self.captureEvent.wait()  
		frame = self.capture_frame	
		#while frame == None: # If there are problems, keep retrying until an image can be read.
			# time.sleep(0)
			#with self.capture_lock:	
			#	frame  = self.capture_frame
				#frame = self.processing_frame

        # We are using Motion JPEG, but OpenCV defaults to capture raw images,
        # so we must encode it into JPEG in order to correctly display the
        # video stream.
		ret, jpeg = cv2.imencode('.jpg', frame)
		#cv2.imwrite("stream/frame.jpg", frame)
		#self.previous_frame = frame
		return jpeg.tostring()

    def read_frame(self):
		#frame = None
		#with self.capture_lock:
		capture_blocker = self.captureEvent.wait()  
		frame = self.capture_frame	
		#while frame == None: # If there are problems, keep retrying until an image can be read.
		
		#	with self.capture_lock:	
		#		frame  = self.capture_frame
		#time.sleep(1.0/CAPTURE_HZ)

		return frame

    def read_processed(self):
		frame = None
		with self.capture_lock:
			frame = self.processing_frame	
		while frame == None: # If there are problems, keep retrying until an image can be read.
			#time.sleep(0)
			with self.capture_lock:	
				frame = self.processing_frame

        # We are using Motion JPEG, but OpenCV defaults to capture raw images,
        # so we must encode it into JPEG in order to correctly display the
        # video stream.
		ret, jpeg = cv2.imencode('.jpg', frame)
		return jpeg.tostring()

