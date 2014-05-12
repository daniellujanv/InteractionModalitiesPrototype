import cv2
import numpy as np
import HandGestureRecognition as HGRecon


def getImageSequence(capture, fastForward):
    '''Load the video sequence (fileName) and proceeds, fastForward number of frames.'''
    global frameNumber
   
    for t in range(fastForward):
        isSequenceOK, originalImage = capture.read()  # Get the first frames
        frameNumber = frameNumber+1
    return originalImage, isSequenceOK

def printUsage():
    print "Q or ESC: Stop"
    print "SPACE: Pause"
    
def run(speed,video): 
    global image, repeat
    '''MAIN Method to load the image sequence and handle user inputs'''   
    #--------------------------------video
    setupWindowSliders()
    capture = cv2.VideoCapture(video)
    tempSpeed = 1
    while repeat:
        image, isSequenceOK = getImageSequence(capture,speed)
        image = cv2.resize(image, size1)
        #print isSequenceOK
        #H,W,_ = image.shape
        #record = False
        if(isSequenceOK):
            processImage(image)
            printUsage()
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
                processImage(image)
                image, isSequenceOK = getImageSequence(capture,speed)



def processImage(image):
    sliderVals = getSliderVals()
    processed = HGRecon.handRecognition(image, sliderVals["Hmax"], sliderVals["Hmin"], sliderVals["Smax"], sliderVals["Smin"])

    cv2.imshow("Result", processed)
    #cv2.imshow("Details", image)

def setupWindowSliders():
    ''' Define windows for displaying the results and create trackbars'''
    cv2.namedWindow("Result")
    cv2.namedWindow('Threshold')
    cv2.namedWindow("Details")
    cv2.resizeWindow('Threshold', 400, 30)
    #Threshold value for the pupil intensity
    #Hue 0 - 179
    #Sat 0 - 255
    cv2.createTrackbar('Hue min','Threshold', 1, 179, onSlidersChange)
    cv2.createTrackbar('Sat min','Threshold', 1, 255, onSlidersChange)
    cv2.createTrackbar('Hue max','Threshold', 120, 179, onSlidersChange)
    cv2.createTrackbar('Sat max','Threshold', 50, 255, onSlidersChange)
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

size1 = (450,350)
frameNumber = 0
speed = 1
repeat = True

video = 'videos/testHandRecon.mov'
run(speed, video)