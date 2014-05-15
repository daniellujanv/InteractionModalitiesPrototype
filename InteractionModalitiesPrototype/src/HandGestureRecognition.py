import cv2
import numpy as np
from Tools import RegionProps
import Tools as tools

global regionProps
regionProps = RegionProps()

'''
-------------------------------
- image in HSV
- restrict channels
- biggest contour as hand
'''
def handRecognition(img, sliderVals, gestures):
    imgColor = img.copy()
    img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    
    minRange = cv2.cv.Scalar(sliderVals['Hmin'] , sliderVals['Smin'], 0.0)
    maxRange = cv2.cv.Scalar(sliderVals['Hmax'], sliderVals['Smax'], 255.0)
    img = cv2.inRange(img, minRange, maxRange)
    img = 255 - img
    kernel = np.ones((9, 9), np.uint8)
    img = cv2.dilate(cv2.erode(img, kernel), kernel)
    cv2.imshow("Details", img)
    #tools.showImages(testing=img)
    contours, _ = cv2.findContours(img, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    #biggestContour[0] == contour, biggestContour[1] == props
    if(len(contours) > 0):
        biggestContour = contours[0]
        biggestContourProps = regionProps.CalcContourProperties(contours[0], ["Area", "Centroid"])
        for contour in contours:
            #print contour
            props = regionProps.CalcContourProperties(contour, ["Area", "Centroid"])
            if(biggestContourProps["Area"] < props["Area"]):        
                biggestContour = contour
                biggestContourProps = props
        epsilon = cv2.arcLength(biggestContour, True)*0.0025
        biggestContour = cv2.approxPolyDP(biggestContour, epsilon, True)
        center = (int(biggestContourProps["Centroid"][0]), int(biggestContourProps["Centroid"][1]))
        cv2.circle(imgColor, center, 5, (0, 0, 255), (int(biggestContourProps["Area"]*.00005) + 1) ) 
        #cv2.drawContours(imgColor, [biggestContour], -1, (0, 255, 0))
        convexhull = cv2.convexHull(biggestContour, returnPoints = False)
        defects = cv2.convexityDefects(biggestContour, convexhull)
        newGestures = detectGestures(imgColor, biggestContour, biggestContourProps, defects, gestures)
        #cv2.drawContours(imgColor, [convexhull], -1, (0, 0, 255))

    return imgColor, newGestures

def detectGestures(imgColor, contour, contourProps, defects, gestures):
    
    if(gestures['Init'] == False):
        defects = removeDefects(defects, contour, contourProps, 'Init')
        positiveDefects = 0
        positiveDefects = detectInitGesture(defects, contour, contourProps)
        if(positiveDefects >= 4):
            gestures['Init'] = True
            print "Gesture Detected: INIT"
            drawDefects(imgColor, defects, contour, contourProps)
            return gestures #return at this point so no other gesture can start
            
    else:# gesture INIT already detected, find other gestures
        #once INIT = True, we can always stop/end interaction
        gestures['End'] = False
        defects = removeDefects(defects, contour, contourProps, 'End')
        positiveDefects = 0
        positiveDefects = detectEndGesture(defects, contour, contourProps)
        #if End is detected then we start again
        if(positiveDefects >= 4):
            print "Gesture Detected: END"
            gestures['Init'] = False
            gestures['Rotate_Init'] = False
            gestures['End'] = False
            drawDefects(imgColor, defects, contour, contourProps)    
            return gestures #return at this point so no other gesture can start
        
        if((gestures['Rotate_Init'] == False) and (gestures['Resize_Init'] == False)):
            positiveDefects = 0
            if(positiveDefects >= 4):
                print "Gesture Detected: Rotate_INIT"
                gestures['Rotate_Init'] = True
                drawDefects(imgColor, defects, contour, contourProps)    
                return gestures #return at this point so no other gesture can start
        elif((gestures['Rotate_Init'] == True) and (gestures['Rotate_End'] == False)):
            positiveDefects = 0
            if(positiveDefects >= 4):#second rotation gesture detected
                print "Gesture Detected: Rotate_END"
                gestures['Rotate_Init'] = False
                gestures['Rotate_End'] = False
                drawDefects(imgColor, defects, contour, contourProps)    
                return gestures #return at this point so no other gesture can start
            
        if((gestures['Resize_Init'] == False) and (gestures['Rotate_Init'] == False)):#initial rotate gesture detected, now look for second rotation gesture 
            positiveDefects = 0
            if(positiveDefects >= 4):
                print "Gesture Detected: Resize_INIT"
                gestures['Resize_Init'] = True
                drawDefects(imgColor, defects, contour, contourProps)    
                return gestures #return at this point so no other gesture can start
        elif((gestures['Resize_Init'] == True) and (gestures['Resize_End'] == False)):
            positiveDefects = 0
            if(positiveDefects >= 4):#second rotation gesture detected
                print "Gesture Detected: Resize_END"
                gestures['Resize_Init'] = False
                gestures['Resize_End'] = False
                drawDefects(imgColor, defects, contour, contourProps)    
                return gestures #return at this point so no other gesture can start
        
    drawDefects(imgColor, defects, contour, contourProps)    
    return gestures

def detectInitGesture(defects, contour, contourProps):
    positiveDefects = 0
    for defect in defects:
            start_i, end_i, far_i, distancePointHull = defect[0]
            distancePointHull = round(distancePointHull/256.0)
            end = tuple(contour[end_i][0])
            #line from center of contour to convexhull point
            center = (int(contourProps["Centroid"][0]), int(contourProps["Centroid"][1]))
            # relations of gesture
            distanceCenterHull = tools.getDistanceBetweenPoints(center, end)
            relationCenterHull_EndHull = (distanceCenterHull/distancePointHull)
            relationCenterHull_ClosestPerimeter = (distanceCenterHull/cv2.pointPolygonTest(contour, center, True))
            if( (relationCenterHull_EndHull > 10.0) and (relationCenterHull_ClosestPerimeter > 2.0)):
                positiveDefects = positiveDefects + 1
        #x,y = 10,200
        #msg = "PosDefects: " + str(positiveDefects)
        #tools.setText(imgColor, (x,y), msg)
    return positiveDefects

def detectEndGesture(defects, contour, contourProps):
    positiveDefects = 0
    for defect in defects:
        start_i, end_i, far_i, distancePointHull = defect[0]
        distancePointHull = round(distancePointHull/256.0)
        end = tuple(contour[end_i][0])
    
        #line from center of contour to convexhull point
        center = (int(contourProps["Centroid"][0]), int(contourProps["Centroid"][1]))
        # relations of gesture
        distanceCenterHull = tools.getDistanceBetweenPoints(center, end)
        relationCenterHull_EndHull = (distanceCenterHull/distancePointHull)
        relationCenterHull_ClosestPerimeter = (distanceCenterHull/cv2.pointPolygonTest(contour, center, True))
        if( (relationCenterHull_EndHull > 10.0) and (relationCenterHull_ClosestPerimeter <= 2.0)):
            positiveDefects = positiveDefects + 1
    return positiveDefects

def removeDefects(defects, contour, contProps, gesture):
    finalDefects = []
    if(gesture == 'Init'):
        try:
            for defect in defects:
                start_i, end_i, far_i, distance = defect[0]
                distance = round(distance/256.0)
                start = tuple(contour[start_i][0])
                end = tuple(contour[end_i][0])
                far = tuple(contour[far_i][0])
                #if end[y] < centroid[y] include
                #top left screen x,y = 0,0
                if((end[1] < contProps["Centroid"][1]) and (far[1] < contProps["Centroid"][1])):
                    finalDefects.append(defect)
                #convexhull
        except TypeError, _:
            print "No defects: removeDefects"
    else:
        try:
            for defect in defects:
                start_i, end_i, far_i, distance = defect[0]
                distance = round(distance/256.0)
                start = tuple(contour[start_i][0])
                end = tuple(contour[end_i][0])
                far = tuple(contour[far_i][0])
                #if end[y] < centroid[y] include
                #top left screen x,y = 0,0
                if((end[1] < contProps["Centroid"][1]) and (far[1] < contProps["Centroid"][1])):
                    finalDefects.append(defect)
                #convexhull
        except TypeError, _:
            print "No defects: removeDefects"
    #end if(gesture == 'Init')
    return finalDefects
    
def drawDefects(image, defects, contour, contourProps):
    #try:
       
    center = (int(contourProps["Centroid"][0]), int(contourProps["Centroid"][1]))
    for defect in defects:
        start_i, end_i, far_i, distance = defect[0]
        distance = round(distance/256.0)
        start = tuple(contour[start_i][0])
        end = tuple(contour[end_i][0])
        far = tuple(contour[far_i][0])
        #convexhull
        cv2.line(image, start, end, [0,0,255], 1)
        #line from farthest point to convexhull
        cv2.line(image, far, end, [0,0,255], 1)
        #line from center of contour to convexhull point
        cv2.line(image, end, center, [255,0,255], 1)
        #points
        cv2.circle(image, end, 5, [0,0,255], -1)
        cv2.circle(image, far, 5, [0,0,255], -1)
        #write distance between hull and farthest point
        tools.setText(image, end, str(distance))
        distanceCenterHull = tools.getDistanceBetweenPoints(center, end)
        centerLine = tools.getMidPointInLine(center, end)
        tools.setText(image, centerLine, str(distanceCenterHull))
    #x,y = 10,120
    #msg = "Center-Cont: " + str(round(cv2.pointPolygonTest(contour, center, True)))
    #tools.setText(image, (x,y), msg)
    #except TypeError, _:
    #    print "No defects: drawDefects"


