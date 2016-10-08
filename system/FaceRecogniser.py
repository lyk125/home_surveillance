# FaceRecogniser
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


import cv2
import numpy as np
import os
import glob
import dlib
import sys
import argparse
from PIL import Image
import pickle
import math
import datetime

import threading
import logging

from sklearn.decomposition import PCA
from sklearn.grid_search import GridSearchCV
from sklearn.manifold import TSNE
from sklearn.svm import SVC

import time
start = time.time()
from operator import itemgetter
from datetime import datetime, timedelta

from operator import itemgetter
from sklearn.preprocessing import LabelEncoder
import atexit
from subprocess import Popen, PIPE
import os.path

import numpy as np
np.set_printoptions(precision=2)
import pandas as pd

import aligndlib

import openface

fileDir = os.path.dirname(os.path.realpath(__file__))
luaDir = os.path.join(fileDir, '..', 'batch-represent')
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

class Face_Recogniser(object):
    
    def __init__(self):


		self.net = openface.TorchNeuralNet(args.networkModel, imgDim=args.imgDim,cuda=args.cuda)
		self.align = openface.AlignDlib(args.dlibFacePredictor)
		self.neuralNet_lock = threading.Lock()
		self.predictor = dlib.shape_predictor(args.dlibFacePredictor)

		with open("generated-embeddings/classifier.pkl", 'r') as f: # le = labels, clf = classifier
			(self.le, self.clf) = pickle.load(f) # loads labels and classifier SVM or GMM
    
    def make_prediction(self,rgbFrame,bb):

		logging.debug('\n\n////////////////////// TRYING TO ALIGN FACE //////////////////////\n\n')
		landmarks = self.align.findLandmarks(rgbFrame, bb)
		alignedFace = self.align.align(args.imgDim, rgbFrame, bb,landmarks=landmarks,landmarkIndices=openface.AlignDlib.OUTER_EYES_AND_NOSE)                                                     
		if alignedFace is None:  
		    print("//////////////////////  FACE COULD NOT BE ALIGNED  //////////////////////////")
		    return None

		print("\n//////////////////////  FACE ALIGNED  ////////////////////// \n")
		with self.neuralNet_lock:
		    logging.debug('\n\n////////////////////// TRYING TO RECOGNISE FACE //////////////////////\n\n')
		    persondict = self.recognize_face(alignedFace)
		    logging.debug('\n\n////////////////////// HELLO THERE RECOGNITION //////////////////////\n\n')

		if persondict is None:
		    print("\n//////////////////////  FACE COULD NOT BE RECOGNIZED  //////////////////////////\n")
		    return persondict, alignedFace
		else:
		    print("\n//////////////////////  FACE RECOGNIZED  ////////////////////// \n")
		    return persondict, alignedFace

    def recognize_face(self,img):

	    if self.getRep(img) is None:  
	        return None
	    print("\n//////////////////////  GETTING REPRESENTATION  ////////////////////// \n")
	    rep = self.getRep(img).reshape(1, -1) # gets embedding representation of image
	    start = time.time()
	    predictions = self.clf.predict_proba(rep).ravel() #Computes probabilities of possible outcomes for samples in classifier(clf).

	    maxI = np.argmax(predictions)
	    person1 = self.le.inverse_transform(maxI)
	    confidence1 = int(math.ceil(predictions[maxI]*100))

	    max2 = np.argsort(predictions)[-3:][::-1][1]
	    person2 = self.le.inverse_transform(max2)
	    confidence2 = int(math.ceil(predictions[max2]*100))

	    print("Recognition took {} seconds.".format(time.time() - start))
	    print("Recognized {} with {:.2f} confidence.".format(person1, confidence1))
	    print("Second highest person: {} with {:.2f} confidence.".format(person1, confidence1))    

	    # if isinstance(clf, GMM):
	    #     dist = np.linalg.norm(rep - clf.means_[maxI])
	    #     print("  + Distance from the mean: {}".format(dist) + " " + persondict)

	    persondict = {'name': person1, 'confidence': confidence1}
	    return persondict

    def getRep(self,alignedFace):

	    bgrImg = alignedFace
	    if bgrImg is None:
	        print("unable to load image")
	        return None

	    alignedFace = cv2.cvtColor(bgrImg, cv2.COLOR_BGR2RGB)
	    start = time.time()
	    print("\n//////////////////////  NEURAL NET FORWARD PASS  ////////////////////// \n")
	    rep = self.net.forward(alignedFace)
	    #print("Neural network forward pass took {} seconds.".format(  time.time() - start))
	    return rep

    def reloadClassifier(self):
		with open("generated-embeddings/classifier.pkl", 'r') as f:
			(self.le, self.clf) = pickle.load(f) # loads labels and classifier SVM or GMM
		return True

    def trainClassifier(self):
		path = fileDir + "/aligned-images/cache.t7" 
		try:
		  os.remove(path) # remove cache from aligned images folder
		except:
		  print "Tried to remove cache.t7"
		  pass

		start = time.time()
		aligndlib.alignMain("training-images/","aligned-images/","outerEyesAndNose",args.dlibFacePredictor,args.imgDim)
		print("\nAligning images took {} seconds.".format(time.time() - start))
		  
		done = False
		start = time.time()

		done = self.generate_representation()
		   
		if done is True:
		    print("Representation Generation (Classification Model) took {} seconds.".format(time.time() - start))
		    start = time.time()
		    #Train Model
		    self.train("generated-embeddings/","LinearSvm",-1)
		    print("Training took {} seconds.".format(time.time() - start))
		else:
		    print("Generate representation did not return True")


    def generate_representation(self):
		#2 Generate Representation 
		print "\n" + luaDir + "\n"
		self.cmd = ['/usr/bin/env', 'th', os.path.join(luaDir, 'main.lua'),'-outDir',  "generated-embeddings/" , '-data', "aligned-images/"]                 
		if args.cuda:
		    self.cmd.append('-cuda')
		self.p = Popen(self.cmd, stdin=PIPE, stdout=PIPE, bufsize=0)
		outs, errs = self.p.communicate() #wait for process to exit
		# result = self.p.wait()  # wait for subprocess to finish writing to files - labels.csv reps.csv

		def exitHandler():
		    if self.p.poll() is None:
		        print "======================Something went Wrong============================"
		        self.p.kill()
		        return False
		atexit.register(exitHandler) 

		return True


    def train(self,workDir,classifier,ldaDim):
		print("Loading embeddings.")
		fname = "{}/labels.csv".format(workDir) #labels of faces
		labels = pd.read_csv(fname, header=None).as_matrix()[:, 1]
		labels = map(itemgetter(1),
		           map(os.path.split,
		               map(os.path.dirname, labels)))  

		fname = "{}/reps.csv".format(workDir) #representations of faces
		embeddings = pd.read_csv(fname, header=None).as_matrix() #get embeddings as a matrix from reps.csv
		self.le = LabelEncoder().fit(labels) # encodes 
		labelsNum = self.le.transform(labels)
		nClasses = len(self.le.classes_)
		print("Training for {} classes.".format(nClasses))

		if classifier == 'LinearSvm':
		   self.clf = SVC(C=1, kernel='linear', probability=True)
		elif classifier == 'GMM':
		   self.clf = GMM(n_components=nClasses)

		if ldaDim > 0:
		  clf_final =  self.clf
		  self.clf = Pipeline([('lda', LDA(n_components=ldaDim)),
		                  ('clf', clf_final)])

		self.clf.fit(embeddings, labelsNum) #link embeddings to labels

		fName = "{}/classifier.pkl".format(workDir)
		print("Saving classifier to '{}'".format(fName))
		with open(fName, 'w') as f:
		  pickle.dump((self.le,  self.clf), f) # creates character stream and writes to file to use for recognition




