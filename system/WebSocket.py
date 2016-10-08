# Web Socket Server
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


# main.py
# from gevent import monkey
# monkey.patch_all()
#import redis
from flask import Flask, render_template, Response, redirect, url_for, request, jsonify, send_file, session, g
from flask.ext.uploads import UploadSet, configure_uploads, IMAGES
import Camera
from flask.ext.socketio import SocketIO,send, emit #Socketio depends on gevent
import SurveillanceSystem
import json
import logging
import threading
import time
from random import random
import os
import sys
import cv2
import psutil

app = Flask(__name__)
app.config['SECRET_KEY'] = os.urandom(24)
socketio = SocketIO(app)


photos = UploadSet('photos', IMAGES)
app.config['UPLOADED_PHOTOS_DEST'] = 'uploads/imgs'
configure_uploads(app, photos)

Home_Surveillance = SurveillanceSystem.Surveillance_System()



thread1 = threading.Thread() 
thread2 = threading.Thread() 
thread3 = threading.Thread() 
thread1.daemon = False
thread2.daemon = False
thread3.daemon = False



# def trace(frame, event, arg):
#     print "%s, %s:%d" % (event, frame.f_code.co_filename, frame.f_lineno)
#     return trace


#sys.settrace(trace)


@app.route('/', methods=['GET','POST'])
def login():
    error = None
    if request.method == 'POST':
        session.pop('user',None) #drops session everytime user tries to login
        if request.form['username'] != 'admin' or request.form['password'] != 'admin':
            error = 'Invalid username or password. Please try again'
        else:
            session['user'] = request.form['username']
            return redirect(url_for('home'))

    return render_template('login.html', error = error)

@app.route('/home')
def home():
    if g.user:
        return render_template('index.html')

    return redirect(url_for('login'))


@app.before_request
def before_request():
    g.user = None
    if 'user' in session:
        g.user = session['user']


@app.route('/upload', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST' and 'photo' in request.files:
        try:
            filename = photos.save(request.files['photo'])
            image = request.files['photo']
            name = request.form.get('name')
            image = cv2.imread('uploads/imgs/' + filename)
            wriitenToDir = Home_Surveillance.add_face(name,image, upload = True)
            message = "file uploaded successfully"
        except:
             message = "file upload unsuccessfull"

        return render_template('index.html', message = message)
    if g.user:
        return render_template('index.html')
    else:
        return redirect(url_for('login'))

def gen(camera):
    while True:
        frame = camera.read_jpg()   # read_jpg()  #read_processed()    
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')  # builds 'jpeg' data with header and payload


@app.route('/video_streamer/<camNum>')
def video_streamer(camNum):
    return Response(gen(Home_Surveillance.cameras[int(camNum)]),
                    mimetype='multipart/x-mixed-replace; boundary=frame') # a stream where each part replaces the previous part the multipart/x-mixed-replace content type must be used.

def system_monitoring():
    while True:

        camera_processing_fps = []

        for camera in Home_Surveillance.cameras:
    
            camera_processing_fps.append("{0:.2f}".format(camera.processing_fps))
            print "FPS: " +str(camera.processing_fps) + " " + str(camera.streaming_fps) #math.ceil(persondict['confidence']*100
        systemState = {'cpu':cpu_usage(),'memory':memory_usage(), 'processing_fps': camera_processing_fps}
        socketio.emit('system_monitoring', json.dumps(systemState) ,namespace='/test')
        time.sleep(3)

def cpu_usage():
      psutil.cpu_percent(interval=1, percpu=False) #ignore first call - often returns 0
      psutil.cpu_percent(interval=1, percpu=False) 
      cpu_load = psutil.cpu_percent(interval=1, percpu=False)
      print "CPU Load: " + str(cpu_load)  
      return cpu_load  

def memory_usage():
     mem_usage = psutil.virtual_memory().percent
     print "System Memory Usage: " + str( mem_usage) 
     return mem_usage 


@app.route('/add_camera', methods = ['GET','POST'])
def add_camera():
    if request.method == 'POST':  
        camURL = request.form.get('camURL')
        application = request.form.get('application')
        detectionMethod = request.form.get('detectionMethod')
      
        with Home_Surveillance.cameras_lock:
            Home_Surveillance.add_camera(SurveillanceSystem.Camera.VideoCamera(camURL,application,detectionMethod))  
        data = {"camNum": len(Home_Surveillance.cameras) -1}
        return jsonify(data)
    return render_template('index.html')

@app.route('/remove_camera', methods = ['GET','POST'])
def remove_camera():
    if request.method == 'POST':
        alertID = request.form.get('alert_id')

        with Home_Surveillance.alerts_lock:
            for i, alert in enumerate(Home_Surveillance.alerts):
                if alert.id == alertID:
                    del Home_Surveillance.alerts[i]
                    break
           
        data = {"alert_status": "removed"}
        return jsonify(data)
    return render_template('index.html')

@app.route('/create_alert', methods = ['GET','POST'])
def create_alert():
    if request.method == 'POST':
        camera = request.form.get('camera')
        emailAddress = request.form.get('emailAddress')
        event = request.form.get('eventdetail')
        alarmstate = request.form.get('alarmstate')
        person = request.form.get('person')
        push_alert = request.form.get('push_alert')
        email_alert = request.form.get('email_alert')
        trigger_alarm = request.form.get('trigger_alarm')
        notify_police = request.form.get('notify_police')
        confidence = request.form.get('confidence')

        print "unknownconfidence: "+confidence 

        actions = {'push_alert': push_alert , 'email_alert':email_alert , 'trigger_alarm':trigger_alarm , 'notify_police':notify_police}
        with Home_Surveillance.alerts_lock:
            Home_Surveillance.alerts.append(SurveillanceSystem.Alert(alarmstate,camera, event, person, actions, emailAddress, int(confidence)))  #alarmState,camera, event, person, action)
        Home_Surveillance.alerts[-1].id 
        data = {"alert_id": Home_Surveillance.alerts[-1].id, "alert_message": "Alert if " + Home_Surveillance.alerts[-1].alertString}
        return jsonify(data)
    return render_template('index.html')



@app.route('/remove_alert', methods = ['GET','POST'])
def remove_alert():
    if request.method == 'POST':
        alertID = request.form.get('alert_id')

        with Home_Surveillance.alerts_lock:
            for i, alert in enumerate(Home_Surveillance.alerts):
                if alert.id == alertID:
                    del Home_Surveillance.alerts[i]
                    break
           
        data = {"alert_status": "removed"}
        return jsonify(data)
    return render_template('index.html')

@app.route('/remove_face', methods = ['GET','POST'])
def remove_face():
    if request.method == 'POST':
        predicted_name = request.form.get('predicted_name')
        camNum = request.form.get('camera')

       
        with Home_Surveillance.cameras[int(camNum)].people_dict_lock:
            try:   
                # if Home_Surveillance.cameras[int(camNum)].people[predicted_name].identity == "unknown":
                #     Home_Surveillance.Person.person_count -= 1 
                del Home_Surveillance.cameras[int(camNum)].people[predicted_name]  
                print "\n\n\n======================= REMOVED: " + predicted_name + "=========================\n\n\n"
            except Exception as e:
                print "\n\n\nERROR could not remove Face\n\n\n" 
                print e
                pass

        data = {"face_removed":  'true'}
        return jsonify(data)
    return render_template('index.html')

@app.route('/add_face', methods = ['GET','POST'])
def add_face():
    if request.method == 'POST':
        new_name = request.form.get('new_name')
        person_id = request.form.get('person_id')
        camNum = request.form.get('camera')
        img = None
        #Home_Surveillance.cameras[int(camNum)].people[name].thumbnail 
        

        with Home_Surveillance.cameras[int(camNum)].people_dict_lock:  
            try:  
                img = Home_Surveillance.cameras[int(camNum)].people[person_id].face   #gets face of person detected in cameras 
                predicted_name = Home_Surveillance.cameras[int(camNum)].people[person_id].identity
                del Home_Surveillance.cameras[int(camNum)].people[person_id]    #removes face from people detected in all cameras 
            except Exception as e:
                print "\n\n\nERROR could not add Face\n\n\n" + e
 

        if int(new_name) != 1:
            wriitenToDir = Home_Surveillance.add_face(new_name,img, upload = False)
        else:
            wriitenToDir = Home_Surveillance.add_face(predicted_name,img, upload = False)

        systemData = {'camNum': len(Home_Surveillance.cameras) , 'people': Home_Surveillance.peopleDB, 'onConnect': False}
        socketio.emit('system_data', json.dumps(systemData) ,namespace='/test')
           
        data = {"face_added":  wriitenToDir}
        return jsonify(data)
    return render_template('index.html')

@app.route('/retrain_classifier', methods = ['GET','POST'])
def retrain_classifier():
    if request.method == 'POST':
        Home_Surveillance.trainingEvent.clear() # block processing threads
        retrained = Home_Surveillance.recogniser.trainClassifier()
        Home_Surveillance.trainingEvent.set() # release processing threads       
        data = {"finished":  retrained}
        return jsonify(data)
    return render_template('index.html')



@app.route('/get_faceimg/<name>')
def get_faceimg(name):  
    key,camNum = name.split("_")
    try:
        with Home_Surveillance.cameras[int(camNum)].people_dict_lock:
            img = Home_Surveillance.cameras[int(camNum)].people[key].thumbnail 
    except Exception as e:
        print "\n\n\n\nError\n\n\n"
        print e 
        img = ""

    if img == "":
        return "http://www.character-education.org.uk/images/exec/speaker-placeholder.png"            

    return  Response((b'--frame\r\n'
                     b'Content-Type: image/jpeg\r\n\r\n' + img + b'\r\n\r\n'),
                    mimetype='multipart/x-mixed-replace; boundary=frame') #send_file(img, mimetype='image/jpg')


@app.route('/get_all_faceimgs/<name>')
def get_faceimgs(name):  
    key, camNum, imgNum = name.split("_")

    try:
        with Home_Surveillance.cameras[int(camNum)].people_dict_lock:
            img = Home_Surveillance.cameras[int(camNum)].people[key].thumbnails[imgNum] 
    except Exception as e:
        print "\n\n\n\nError\n\n\n"
        print e 
        img = ""

    if img == "":
        return "http://www.character-education.org.uk/images/exec/speaker-placeholder.png"            

    return  Response((b'--frame\r\n'
                     b'Content-Type: image/jpeg\r\n\r\n' + img + b'\r\n\r\n'),
                    mimetype='multipart/x-mixed-replace; boundary=frame') #send_file(img, mimetype='image/jpg')



def update_faces():
     while True:
            peopledata = []
            persondict = {}
            thumbnail = None
            with Home_Surveillance.cameras_lock:
                for i, camera in enumerate(Home_Surveillance.cameras):
                    with Home_Surveillance.cameras[i].people_dict_lock:
                        for key, obj in camera.people.iteritems():  
                            persondict = {'identity': key , 'confidence': obj.confidence, 'camera': i, 'timeD':obj.time, 'prediction': obj.get_prediction(),'thumbnailNum': len(obj.thumbnails)}
                            print persondict
                            peopledata.append(persondict)
         
            socketio.emit('people_detected', json.dumps(peopledata) ,namespace='/test')
            time.sleep(4)

def alarm_state():
     while True:
            alarmstatus = {'state': Home_Surveillance.alarmState , 'triggered': Home_Surveillance.alarmTriggerd }
            socketio.emit('alarm_status', json.dumps(alarmstatus) ,namespace='/test')
            time.sleep(3)


@socketio.on('alarm_state_change', namespace='/test') 
def alarm_state_change():   
    Home_Surveillance.change_alarmState()

@socketio.on('panic', namespace='/test') 
def panic(): 
    Home_Surveillance.trigger_alarm()
   

@socketio.on('my event', namespace='/test') #socketio used to receive websocket messages # Namespaces allow a cliet to open multiple connectiosn to the server that are multiplexed on a single socket
def test_message(message):   #custom events deliver JSON payload 

    emit('my response', {'data': message['data']}) # emit() sends a message under a custom event name

@socketio.on('my broadcast event', namespace='/test')
def test_message(message):
    emit('my response', {'data': message['data']}, broadcast=True) # broadcast=True optional argument all clients connected to the namespace receive the message

                   
@socketio.on('connect', namespace='/test') 
def test_connect(): 
                          #first argumenent is the event name, connect and disconnect are special event names the others are custom events
    global thread1
    global thread2 #need visibility of global thread object
    global thread3

   
    print "\n\nclient connected\n\n"


    if not thread1.isAlive():
        print "Starting Thread1"
        thread1 = threading.Thread(name='alarmstate_process_thread_',target= alarm_state, args=())
        thread1.start()
   
    if not thread2.isAlive():
        print "Starting Thread2"
        thread2 = threading.Thread(name='websocket_process_thread_',target= update_faces, args=())
        thread2.start()

    if not thread3.isAlive():
        print "Starting Thread3"
        thread3 = threading.Thread(name='monitoring_process_thread_',target= system_monitoring, args=())
        thread3.start()

    cameraData = {}
    cameras = []

    with Home_Surveillance.cameras_lock:
        for i, camera in enumerate(Home_Surveillance.cameras):
            with Home_Surveillance.cameras[i].people_dict_lock:
                cameraData = {'camNum': i , 'url': camera.url}
                print cameraData
                cameras.append(cameraData)

    alertData = {}
    alerts = []

    for i, alert in enumerate(Home_Surveillance.alerts):
        with Home_Surveillance.alerts_lock:
            alertData = {'alert_id': alert.id , 'alert_message':  "Alert if " + alert.alertString}
            print alertData
            alerts.append(alertData)
   

    systemData = {'camNum': len(Home_Surveillance.cameras) , 'people': Home_Surveillance.peopleDB, 'cameras': cameras, 'alerts': alerts, 'onConnect': True}
    socketio.emit('system_data', json.dumps(systemData) ,namespace='/test')


    #emit('my response', {'data': 'Connected'})

@socketio.on('disconnect', namespace='/test')
def test_disconnect():
    print('Client disconnected')


if __name__ == '__main__':
#    # app.run(host='0.0.0.0', debug=True)
     socketio.run(app, host='0.0.0.0', debug = True, use_reloader=False) #starts server on default port 5000 and makes socket connection available to other hosts (host = '0.0.0.0')
    
