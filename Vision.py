# importing the necessary packages and libraries
import cv2
import numpy as np
import requests
import dlib
import imutils
import face_recognition
from boltiot import Bolt
from scipy.spatial import distance as dist
import time
import econf
from imutils import face_utils
from imutils.video import FileVideoStream
from imutils.video import VideoStream


#specifying model to be used
prediction_model = "shape_predictor_68_face_landmarks.dat"


# BOLT IOT module key and mailgun id to send email alert
from boltiot import Bolt
api_key = "<Enter your API Key>"
device_id  = "<Enter Device ID>"
mybolt=Bolt(api_key, device_id)


#creating function for calculation of eye-aspect ratio
def eye_aspect_ratio(eye):
    # computing the euclidean distances between the two sets of vertical eye landmarks (x, y)-coordinates
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])

    # computing the euclidean distance between the horizontal eye landmark (x, y)-coordinates
    C = dist.euclidean(eye[0], eye[3])

    # computing the eye-aspect ratio
    ear = (A + B) / (2.0 * C)

    # return the eye-aspect ratio
    return ear


# define two constants, one for the eye aspect ratio to indicate blink and then a second constant for the number of consecutive frames the eye must be below the threshold
EYE_AR_THRESH = 0.3
EYE_AR_CONSEC_FRAMES = 3


# initialize the frame counters and the total number of blinks
COUNTER = 0
TOTAL = 0


# initialize dlib's face detector (HOG-based) and then create the facial landmark predictor
print("Loading...")
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(prediction_model)

# grab the indexes of the facial landmarks for the left and right eye, respectively
(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]


# Read the users data and create face encodings
image = face_recognition.load_image_file("Image(mine).jpg")
mface_encoding = face_recognition.face_encodings(image)[0]

known_names = ['Pratyaksh']
known_encods = [mface_encoding]


# start the video stream thread
print("Starting Video Stream")
fileStream = True
vs = VideoStream(src='<Enter URL To IP Webcam Stream>').start()
time.sleep(1.0)
print("Video Stream Started")
face_locations = []
face_encodings = []
face_names = []


# loop over frames from the video stream
while True:
    # grab the frame from the threaded video file stream, resize it, and convert it to grayscale channels
    frame = vs.read()
    frame = imutils.resize(frame, width=1080)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    small_frame = cv2.resize(rgb, (0, 0), fx=0.25, fy=0.25)

  
    # Find all the faces and face encodings in the (current) video frame
    face_locations = face_recognition.face_locations(small_frame)
    face_encodings = face_recognition.face_encodings(small_frame, face_locations)
    name = "Unknown"
    face_names = []
    for face_encoding in face_encodings:
    
        match = face_recognition.compare_faces(known_encods, face_encoding)

        if match[0]:
             name = known_names[0]

        face_names.append(name)

   
    # detect faces in the grayscale frame
        rects = detector(gray, 0)

    # loop over the face detections
        for rect in rects:
            # determine the facial landmarks for the face region, then convert the facial landmark (x, y)-coordinates to a numpy array
            shape = predictor(gray, rect)
            shape = face_utils.shape_to_np(shape)

            # extract the left and right eye coordinates, then use the coordinates to compute the eye aspect ratio for both eyes
            leftEye = shape[lStart:lEnd]
            rightEye = shape[rStart:rEnd]
            leftEAR = eye_aspect_ratio(leftEye)
            rightEAR = eye_aspect_ratio(rightEye)


            # average the eye aspect ratio together for both eyes
            aspect_ratio = (leftEAR + rightEAR) / 2.0


            # compute the convex hull for the left and right eye, then visualize each of the eyes
            leftEyeHull = cv2.convexHull(leftEye)
            rightEyeHull = cv2.convexHull(rightEye)
            cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
            cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)


            # check to see if the eye aspect ratio is below the blink threshold, and if so, increment the blink frame counter
            if aspect_ratio < EYE_AR_THRESH:
                COUNTER += 1


            # otherwise, the eye aspect ratio is not below the blink threshold
            else:
                # if the eyes were closed for a sufficient number of then increment the total number of blinks
                if COUNTER >= EYE_AR_CONSEC_FRAMES:
                    TOTAL += 1
                # reset the eye frame counter
                COUNTER = 0
            # draw the total number of blinks on the frame along with the computed eye aspect ratio for the frame
            cv2.putText(frame, "NO. OF BLINKS : {}".format(TOTAL), (10, 30),
                        cv2.FONT_HERSHEY_DUPLEX, 0.9, (0, 0, 255), 2)
            cv2.putText(frame, "EYE-ASPECT RATIO : {:.2f}".format(aspect_ratio), (300, 30),
                        cv2.FONT_HERSHEY_DUPLEX, 0.9, (0, 0, 255), 2)


    # Display the results
    for (top, right, bottom, left), name in zip(face_locations, face_names):
            # Scale back up face locations since the frame we detected in was scaled to 1/4 size
            top *= 4
            right *= 4
            bottom *= 4
            left *= 4
            
            
            if face_names[0]!='Unknown':
               while TOTAL>=3:
              
                    # Draw a box around the face
                    cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
                    font = cv2.FONT_HERSHEY_DUPLEX
                     # Draw a label with a name below the face
                    cv2.rectangle(frame, (left, bottom - 35),
                          (right, bottom), (0, 0, 255), cv2.FILLED)
                    
                    cv2.putText(frame, name, (left + 6, bottom - 6),
            
                        font, 1.0, (255, 255, 255), 1)
                    cv2.putText(frame, 'Authorized!!!', (frame.shape[1]//2, frame.shape[0]//2), font, 1.0, (0, 255, 0), 1)
                    break
               
            else:
                font = cv2.FONT_HERSHEY_DUPLEX
                cv2.putText(frame, 'Intruder!!!', (frame.shape[1]//2, frame.shape[0]//2), font, 2.0, (0, 0, 255), 1)
                
                cv2.imwrite('Intruder.jpg',frame[right:right+left,top:top+bottom])
                
                print("\n[CAUTION] Sending alert to Pratyaksh")
                response=mybolt.digitalWrite('0','HIGH')

               
                print("\n [CAUTION] Sending e-mail to Pratyaksh")
                
                # creating mailgun API call
                def send_simple_message():
	                return requests.post(
		               "<https://api.mailgun.net/v3/YOUR_DOMAIN_NAME/messages>",
		                auth=("api", "<Enter Mailgun API Key>"),
		                data={"from": "Excited User <mailgun@YOUR_DOMAIN_NAME>",
			            "to": "<Enter Reciever's Email>",
			            "subject": "Intruder Detected",
			            "text": "Someone tried to access your application",   
                        "html":'<html>Inline image here:<img src="cid:opencv_Intruder.jpg"></html>'})

                response=mybolt.digitalWrite('0','LOW')
    

    # show the frame
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF


    # if the `q` key was pressed, break from the loop
    if key == ord("q"):
        break


# do a bit of cleanup
print("\n Recognition Task Accomplished")
cv2.destroyAllWindows()
vs.stop()
