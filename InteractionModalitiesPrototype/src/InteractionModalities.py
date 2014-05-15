import cv2
import numpy as np
import HandGestureRecognition as HGRecon
import Tools as tools

def run(speed,video): 
    global image, repeat, gestures, frameNumber
    '''MAIN Method to load the image sequence and handle user inputs'''   
    #--------------------------------video
    setupWindowSliders()
    capture = cv2.VideoCapture(video)
    tempSpeed = 1
    while repeat:
        image, isSequenceOK, frameNumber = tools.getImageSequence(capture,speed, frameNumber)
        image = cv2.resize(image, size1)
        #print isSequenceOK
        #H,W,_ = image.shape
        #record = False
        if(isSequenceOK):
            gestures = processImage(image, gestures)
            tools.printUsage()
        #writer = cv2.VideoWriter('CubeProjections.avi', cv.CV_FOURCC('D','I','V','3'), 5.0, (W,H), True)
        while(isSequenceOK):
            #OriginalImage = image.copy()
            image = cv2.resize(image, size1)
            inputKey = cv2.waitKey(1)
            if inputKey == 32:#  stop by SPACE key
                sliderVals = getSliderVals()
                cv2.setTrackbarPos('Stop/Start','Threshold',not sliderVals['Running'])
                #update(OriginalImage)
                if speed==0:     
                    speed = tempSpeed;
                else:
                    tempSpeed=speed
                    speed = 0;                    
            if (inputKey == 27) or (inputKey == ord('q')):#  break by ECS key
                repeat=False
                break    
            #get next sequence and update
            if (speed>0):
                gestures = processImage(image, gestures)
                image, isSequenceOK, frameNumber = tools.getImageSequence(capture,speed, frameNumber)
        #end while(isSequenceOk)
        gestures = {'Init':False,'End':False}
    #end while(repeat)

def processImage(image, gestures):
    sliderVals = getSliderVals()
    processed, newGestures = HGRecon.handRecognition(image, sliderVals, gestures)
    x,y = 10,20
    msg = "HMAX: " + str(sliderVals["Hmax"]) + " HMIN: " + str(sliderVals["Hmin"])
    tools.setText(processed, (x,y), msg)
    y = 40
    msg = "SMAX: " + str(sliderVals["Smax"]) + " SMIN: " + str(sliderVals["Smin"])
    tools.setText(processed, (x,y), msg)
    for gestName in newGestures.keys():
        y = y+20
        msg = gestName+": " + str(newGestures[gestName])
        tools.setText(processed, (x,y), msg)
    cv2.imshow("Result", processed)
    #cv2.imshow("Details", image)
    return newGestures

def setupWindowSliders():
    ''' Define windows for displaying the results and create trackbars'''
    cv2.namedWindow("Result")
    cv2.namedWindow('Threshold')
    cv2.namedWindow("Details")
    cv2.resizeWindow('Threshold', 400, 30)
    #Threshold value for the pupil intensity
    #Hue 0 - 179
    #Sat 0 - 255
    
    cv2.createTrackbar('Hue min','Threshold', 0, 179, onSlidersChange)
    cv2.createTrackbar('Sat min','Threshold', 0, 255, onSlidersChange)
    cv2.createTrackbar('Hue max','Threshold', 134, 179, onSlidersChange)
    cv2.createTrackbar('Sat max','Threshold', 48, 255, onSlidersChange)
    #Value to indicate whether to run or pause the video
    cv2.createTrackbar('Stop/Start','Threshold', 0,1, onSlidersChange)

def getSliderVals():
    '''Extract the values of the sliders and return these in a dictionary'''
    sliderVals={}
    sliderVals['Hmax'] = cv2.getTrackbarPos('Hue max', 'Threshold')
    sliderVals['Hmin'] = cv2.getTrackbarPos('Hue min', 'Threshold')
    sliderVals['Smax'] = cv2.getTrackbarPos('Sat max', 'Threshold')
    sliderVals['Smin'] = cv2.getTrackbarPos('Sat min', 'Threshold')
    sliderVals['Running'] = 1==cv2.getTrackbarPos('Stop/Start', 'Threshold')
    return sliderVals

def onSlidersChange(dummy=None):
    ''' Handle updates when slides have changed.
     This  function only updates the display when the video is put on pause'''
    global image
    sv=getSliderVals()
    if(not sv['Running']): # if pause
        global speed
        speed = 0
        processImage(image)

#---------------------------------- main method -------------------------------------------------#
global frameNumber
global image
global size1
global speed
global repeat
global gestures

size1 = (450,350)
frameNumber = 0
speed = 1
repeat = True
gestures = {'Init':False, 'Rotate_Init':False, 'Rotate_End':False, 'Resize_Init':False, 'Resize_End':False, 'End':False}

video = 'videos/testHandRecon4.mp4'
run(speed, video)