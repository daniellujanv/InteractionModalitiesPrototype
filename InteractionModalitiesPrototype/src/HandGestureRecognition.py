import cv2
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
def handRecognition(img, hmax, hmin, smax, smin):
    imgColor = img.copy()
    img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    
    minRange = cv2.cv.Scalar(hmin , smin, 1.0)
    maxRange = cv2.cv.Scalar(hmax, smax, 255.0)
    img = cv2.inRange(img, minRange, maxRange)
    img = 255 - img
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
    
        center = (int(biggestContourProps["Centroid"][0]), int(biggestContourProps["Centroid"][1]))
        cv2.circle(imgColor, center, 5, (0, 0, 255), (int(biggestContourProps["Area"]*.00005) + 1) ) 
        cv2.drawContours(imgColor, [biggestContour], -1, (0, 255, 0))
        convexhull = cv2.convexHull(biggestContour, returnPoints = False)
        defects = cv2.convexityDefects(biggestContour, convexhull)
        for defect in defects:
            start_i, end_i, far_i, _ = defect[0]
            start = tuple(biggestContour[start_i][0])
            end = tuple(biggestContour[end_i][0])
            far = tuple(biggestContour[far_i][0])
            cv2.line(imgColor, start, end, [0,255,0], 2)
            cv2.circle(imgColor, far, 5, [0,0,255], -1)
        #cv2.drawContours(imgColor, [convexhull], -1, (0, 0, 255))
    
    return imgColor
