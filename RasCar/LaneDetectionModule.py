import cv2
import numpy as np
import utlis
from MotorModule import Motor

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
    if len(curveList) > avgVal:  # chỉ giữ 10 giá trị trong danh sách
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
    if curve > 0.2:
        curve = 1
    elif curve < -0.2:
        curve = -1
    else:
        curve = 0

    return curve


if __name__ == "__main__":
    motor = Motor(6, 13, 12, 16)
    #cap = cv2.VideoCapture("vid1.mp4")
    cap = cv2.VideoCapture(0)
    intalTrackBarVals = [80, 163, 40, 240]
    utlis.initializeTrackbars(intalTrackBarVals)
    frameCounter = 0

    while True:
        
        # lặp lại video
        #frameCounter += 1
        #if cap.get(cv2.CAP_PROP_FRAME_COUNT) == frameCounter:
            #cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            #frameCounter = 0

        success, img = cap.read()  # GET THE IMAGE
        img = cv2.resize(img, (480, 240))  # RESIZE
        cv2.imshow("Vid", img)
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
