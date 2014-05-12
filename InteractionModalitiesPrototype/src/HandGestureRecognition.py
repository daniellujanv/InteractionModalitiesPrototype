import cv2
import Tools as tools
from Tools import RegionProps

global regionProps
regionProps = RegionProps()

'''
-------------------------------
- image in HSV
- restrict channels
- biggest contour as hand
'''
def approach2(img):
    imgColor = img.copy()
    img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    minRange = cv2.cv.Scalar(1.0 , 94.0, 1.0)
    maxRange = cv2.cv.Scalar(29.0, 255.0, 255.0)
    img = cv2.inRange(img, minRange, maxRange)
    #tools.showImages(testing=img)
    contours, _ = cv2.findContours(img, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    #biggestContour[0] == contour, biggestContour[1] == props
    biggestContour = contours[0]
    biggestContourProps = regionProps.CalcContourProperties(contours[0], ["Area", "Boundingbox", "Centroid", "Extend"])
    for contour in contours:
        #print contour
        props = tools.RegionProps.CalcContourProperties(contour, ["Area", "Boundingbox", "Centroid", "Extend"])
        if(biggestContourProps["Area"] < props["Area"]):        
            biggestContour = contour
            biggestContourProps = props

    center = (int(biggestContourProps["Centroid"][0]), int(biggestContourProps["Centroid"][1]))
    cv2.circle(imgColor, center, 5, (0, 0, 255), (int(biggestContourProps["Area"]*.00005) + 1) ) 
    cv2.drawContours(imgColor, [biggestContour], -1, (0, 255, 0))
    hull =  cv2.convexHull(biggestContour)
    cv2.drawContours(imgColor, [hull], -1, (0, 0, 255))
    
    return imgColor
