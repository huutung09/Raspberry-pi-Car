from MotorModule import Motor
from LaneDetectionModule import getLaneCurve
import WebcamModule
import utlis
import cv2
import queue
import threading 
from classification import onPredictSign


motor = Motor(6, 13, 12, 16)
#cap = cv2.VideoCapture("vid1.mp4")
cap = cv2.VideoCapture(0)
intalTrackBarVals = [80, 163, 40, 240]
utlis.initializeTrackbars(intalTrackBarVals)
frameCounter = 0
q = queue.Queue()
t = threading.Thread(target=onPredictSign, args=(q, cap))
t.start()
while True:
    
    # lặp lại video
    #frameCounter += 1
    #if cap.get(cv2.CAP_PROP_FRAME_COUNT) == frameCounter:
        #cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        #frameCounter = 0

    success, img = cap.read()  # GET THE IMAGE
    img = cv2.resize(img, (480, 240))  # RESIZE 
    cv2.imshow("Vid", img)
    sign = q.get()
    print(sign)
    curve = getLaneCurve(img)
    print(curve)
    motor.motorB_forward(50)
    if curve == -1:
        # left
        motor.motorA_left()
    elif curve == 1:
        #right
        motor.motorA_right()
    else:
        #forward
        motor.motorA_forward()
    cv2.waitKey(1)
