import cv2
import numpy as np
import HandGestureRecognition


def getImageSequence(capture, fastForward):
    '''Load the video sequence (fileName) and proceeds, fastForward number of frames.'''
    global frameNumber
   
    for t in range(fastForward):
        isSequenceOK, originalImage = capture.read()  # Get the first frames
        frameNumber = frameNumber+1
    return originalImage, isSequenceOK

def run(speed,video): 
    
    '''MAIN Method to load the image sequence and handle user inputs'''   

    #--------------------------------video
    capture = cv2.VideoCapture(video)
    tempSpeed = 1

    image, isSequenceOK = getImageSequence(capture,speed)
    print isSequenceOK
    H,W,_ = image.shape
    record = False

    if(isSequenceOK):
        update(image)
        #printUsage()
    
    #writer = cv2.VideoWriter('CubeProjections.avi', cv.CV_FOURCC('D','I','V','3'), 5.0, (W,H), True)
    while(isSequenceOK):
        OriginalImage = image.copy()
     
        
        inputKey = cv2.waitKey(1)
        
        if inputKey == 32:#  stop by SPACE key
            update(OriginalImage)
            if speed==0:     
                speed = tempSpeed;
            else:
                tempSpeed=speed
                speed = 0;                    
            
        if (inputKey == 27) or (inputKey == ord('q')):#  break by ECS key
            break    
                
        if inputKey == ord('p') or inputKey == ord('P'):
            global ProcessFrame
            if ProcessFrame:     
                ProcessFrame = False;
                
            else:
                ProcessFrame = True;
            update(OriginalImage)

def update(image):
    
    cv2.imshow("Result", image)

#---------------------------------- main method -------------------------------------------------#
video = 'videos/startStop'
run(1, video)