from typing import ClassVar


import importlib.util
import os
import numpy as np
import time
import cv2

from VideoStream import VideoStream
from tensorflow.lite.python.interpreter import Interpreter
from threading import Thread


class ObjectDetection:
    def __init__(self):
        self.stopped = False
        self.imput_mean = 127.5
        self.input_std = 127.5
        self.frame_rate_calc = 1

        self.MODEL_NAME = "Sample_TFLite_model"
        self.GRAPH_NAME = "detect.tflite"
        self.LABELMAP_NAME = "labelmap.txt"
        self.min_conf_threshold = 0.5
        self.resW = "1280"
        self.resH = "720"
        self.imW, self.imH = int(self.resW), int(self.resH)
        self.CWD_PATH = os.getcwd()

        # Path to .tflite file, which contains the model that is used for object detection
        self.PATH_TO_CKPT = os.path.join(self.CWD_PATH,self.MODEL_NAME,self.GRAPH_NAME)

        # Path to label map file
        self.PATH_TO_LABELS = os.path.join(self.CWD_PATH,self.MODEL_NAME,self.LABELMAP_NAME)

        # Load the label map
        with open(self.PATH_TO_LABELS, 'r') as f:
            self.labels = [line.strip() for line in f.readlines()]
        # Have to do a weird fix for label map if using the COCO "starter model" from
        # https://www.tensorflow.org/lite/models/object_detection/overview
        # First label is '???', which has to be removed.
        if self.labels[0] == '???':
            del(self.labels[0])

        #load the tensorflow lite model
        self.interpreter = Interpreter(model_path=self.PATH_TO_CKPT)

        
        
    def setUp(self):
        print("setting up")
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        self.height = self.input_details[0]['shape'][1]
        self.width = self.input_details[0]['shape'][2]
        self.floating_model = (self.input_details[0]['dtype'] == np.float32)
        self.freq = cv2.getTickFrequency()
        self.videostream = VideoStream(resolution=(self.imW, self.imH), framerate=30).start()
        time.sleep(1)

    def run(self):
        while True:
            t1 = cv2.getTickCount()

            # Grab frame from video stream
            frame1 = self.videostream.read()

            # Acquire frame and resize to expected shape [1xHxWx3]
            frame = frame1.copy()
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_resized = cv2.resize(frame_rgb, (self.width, self.height))
            input_data = np.expand_dims(frame_resized, axis=0)

            # Normalize pixel values if using a floating model (i.e. if model is non-quantized)
            if self.floating_model:
                input_data = (np.float32(input_data) - self.input_mean) / self.input_std

                # Perform the actual detection by running the model with the image as input
            self.interpreter.set_tensor(self.input_details[0]['index'],input_data)
            self.interpreter.invoke()
            # Retrieve detection results
            boxes = self.interpreter.get_tensor(self.output_details[0]['index'])[0] # Bounding box coordinates of detected objects
            classes = self.interpreter.get_tensor(self.output_details[1]['index'])[0] # Class index of detected objects
            scores = self.interpreter.get_tensor(self.output_details[2]['index'])[0] # Confidence of detected objects
            #num = interpreter.get_tensor(output_details[3]['index'])[0]  # Total number of detected objects (inaccurate and not needed)

            # Loop over all detections and draw detection box if confidence is above minimum threshold
            for i in range(len(scores)):
                if ((scores[i] > self.min_conf_threshold) and (scores[i] <= 1.0)):
                    # Get bounding box coordinates and draw box
                    # Interpreter can return coordinates that are outside of image dimensions, need to force them to be within image using max() and min()
                    ymin = int(max(1,(boxes[i][0] * self.imH)))
                    xmin = int(max(1,(boxes[i][1] * self.imW)))
                    ymax = int(min(self.imH,(boxes[i][2] * self.imH)))
                    xmax = int(min(self.imW,(boxes[i][3] * self.imW)))
                    
                    cv2.rectangle(frame, (xmin,ymin), (xmax,ymax), (10, 255, 0), 2)

                    # Draw label
                    object_name = self.labels[int(classes[i])] # Look up object name from "labels" array using class index
                    label = '%s: %d%%' % (object_name, int(scores[i]*100)) # Example: 'person: 72%'
                    labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2) # Get font size
                    label_ymin = max(ymin, labelSize[1] + 10) # Make sure not to draw label too close to top of window
                    cv2.rectangle(frame, (xmin, label_ymin-labelSize[1]-10), (xmin+labelSize[0], label_ymin+baseLine-10), (255, 255, 255), cv2.FILLED) # Draw white box to put label text in
                    cv2.putText(frame, label, (xmin, label_ymin-7), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2) # Draw label text
            # Draw framerate in corner of frame
            cv2.putText(frame,'FPS: {0:.2f}'.format(self.frame_rate_calc),(30,50),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,0),2,cv2.LINE_AA)

            # All the results have been drawn on the frame, so it's time to display it.
            cv2.imshow('Object detector', frame)

            # Calculate framerate
            t2 = cv2.getTickCount()
            time1 = (t2-t1)/self.freq
            self.frame_rate_calc= 1/time1
            if cv2.waitKey(1) == ord('q'):
                break


    def startDetection(self):
        Thread(target=self.run, args=()).start()
        return self

    def read(self):
        return "[OBJECT DETECTED]"


            

test = ObjectDetection()
test.setUp()
test.startDetection()
