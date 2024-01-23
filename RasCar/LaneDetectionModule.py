import cv2
import numpy as np
import utlis
from MotorModule import Motor
import queue
import threading 
from classification import onPredictSign, SVM, localization
from time import sleep
from classification import training

curveList = []
avgVal = 10

def getLaneCurve(img, display=2):

    imgCopy = img.copy()
    imgResult = img.copy()

    ### STEP 1
    imgThres = utlis.thresholding(img)

    #### STEP 2
    hT, wT, c = img.shape  # lấy giá trị chiều cao và độ rộng của img
    points = utlis.valTrackbars()
    imgWarp = utlis.warpImg(imgThres, points, wT, hT)
    imgWarpPoints = utlis.drawPoints(imgCopy, points)

    #### STEP 3
    middlePoint, imgHist = utlis.getHistogram(imgWarp, display=True, minPer=0.5, region=4)
    curveAveragePoint, imgHist = utlis.getHistogram(imgWarp, display=True, minPer=0.9)
    curveRaw = curveAveragePoint - middlePoint

    #### STEP 4
    curveList.append(curveRaw)
    if len(curveList) > avgVal:  # giá trị trong danh sách
        curveList.pop(0)
    curve = int(sum(curveList) / len(curveList))

    #### STEP 5
    if display != 0:
        imgInvWarp = utlis.warpImg(imgWarp, points, wT, hT, inv=True)
        imgInvWarp = cv2.cvtColor(imgInvWarp, cv2.COLOR_GRAY2BGR)
        imgInvWarp[0:hT // 3, 0:wT] = 0, 0, 0
        imgLaneColor = np.zeros_like(img)
        imgLaneColor[:] = 0, 255, 0
        imgLaneColor = cv2.bitwise_and(imgInvWarp, imgLaneColor)
        imgResult = cv2.addWeighted(imgResult, 1, imgLaneColor, 1, 0)
        midY = 450
        cv2.putText(imgResult, str(curve), (wT // 2 - 80, 85), cv2.FONT_HERSHEY_COMPLEX, 2, (255, 0, 255), 3)
        cv2.line(imgResult, (wT // 2, midY), (wT // 2 + (curve * 3), midY), (255, 0, 255), 5)
        cv2.line(imgResult, ((wT // 2 + (curve * 3)), midY - 25), (wT // 2 + (curve * 3), midY + 25), (0, 255, 0), 5)
        for x in range(-30, 30):
            w = wT // 20
            cv2.line(imgResult, (w * x + int(curve // 50), midY - 10),
                     (w * x + int(curve // 50), midY + 10), (0, 0, 255), 2)
        # fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer);
        # cv2.putText(imgResult, 'FPS ' + str(int(fps)), (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (230, 50, 50), 3);
    if display == 2:
        imgStacked = utlis.stackImages(0.7, ([img, imgWarpPoints, imgWarp],
                                             [imgHist, imgLaneColor, imgResult]))
        cv2.imshow('ImageStack', imgStacked)
    elif display == 1:
        cv2.imshow('Resutlt', imgResult)

    # cv2.imshow("Thres", imgThres)
    # cv2.imshow("Warp", imgWarp)
    # cv2.imshow("WarpPoint", imgWarpPoints)
    # cv2.imshow("Histogram", imgHist)

    ## Chuẩn hóa số liệu
    curve /= 100  # Tính %
    if curve > 0.15:
        curve = 1
    elif curve < -0.15:
        curve = -1
    else:
        curve = 0

    return curve


if __name__ == "__main__":
    # motor
    motor = Motor(13, 12, 16)
    #cap = cv2.VideoCapture("vid1.mp4")
    cap = cv2.VideoCapture(0)
    intalTrackBarVals = [80, 163, 20, 240]
    utlis.initializeTrackbars(intalTrackBarVals)
    
#    q = queue.Queue()
#    t = threading.Thread(target=onPredictSign, args=(q, cap))
#    t.start()
    model = training()

    
    
    motor.motorB_start();
    flag = True

    while True:
        success,frame = cap.read()  # GET THE IMAGE
        if not success:
            print("FINISHED")
            break
        width = frame.shape[1]
        height = frame.shape[0]
        frame = cv2.resize(frame, (480, 240))  # RESIZE
        
        curve = getLaneCurve(frame)
#        coordinate, image, sign_type, text = localization(frame,min_size_components, similitary_contour_with_circle, model)
#        if (sign_type > 0):
#            print(SIGNS[sign_type])
        sign = onPredictSign(cap, model)
#        sign = q.get()
        
        
#        print(sign)
        
#        print(curve)
        if flag:
            motor.motorB_forward(35)
        if curve == -1:
            print("left")
            motor.motorA_left()
        elif curve == 1:
            print("right")
            motor.motorA_right()
        else:
            print("forward")
            motor.motorA_forward()
            
        if (sign == 1):
            motor.stop_all()
            print(sign)
            flag = False
        elif (sign == 6):
            motor.motorB_start()
            #motor.motorA_start()
            print(sign)
            flag = True
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            motor.stop_all()
            motor.gpip_cleanup()
            cv2.destroyAllWindows()
            break
