# Home Surveilance with Facial Recognition. 
---

# Installation
---

1) Pull Docker Image

```
docker pull bjoffe/openface_flask
```

2) Run Docker image, make sure you mount your User directory as a volume so you can access your local files

```
docker run -v /Users/:/host/Users -p 9000:9000 -p 8000:8000 -p 5000:5000 -t -i bjoffe/openface_flask  /bin/bash

```

# Usage
---

- Navigate to the home_surveillance project inside the Docker container
- Move into the system directory
- Include dependencies not installed on docker image
```
pip install psutil
pip install flask-uploads
```
```
cd system
```
- Run WebSocket.py
```
python WebSocket.py
```
- Visit ```localhost:5000 ```
- Login Username: ```admin``` Password ```admin```

# Notes and Features
---

- By default the system processes 3 videos
- To add your own IP camera simply add the URL of the camera into field on the camera panel on the client dashboard. 
- Faces that are detected are shown in the faces detected panel on the Dashboard
- To add faces to the database add a folder of images with the name of the person and retrain the classifier by selecting the retrain database on the client dashboard.
- The Dashboard allows you to configure your own email and and alarm trigger alerts.
- The alarm control panel sends http post requests to a web server on a Raspberry PI to control GPIO pins. 
- This project is being developed for the purpose of my thesis and I hope to have a fully functional system by the end of October 2016.
- Currently there are a few bugs and the code is not well commented.

# References
---

- Video Streaming Code - http://www.chioka.in/python-live-video-streaming-example/
- Flask Web Server GPIO - http://mattrichardson.com/Raspberry-Pi-Flask/
- Openface Project - https://cmusatyalab.github.io/openface/
- Flask Websockets - http://blog.miguelgrinberg.com/post/easy-websockets-with-flask-and-gevent

 
