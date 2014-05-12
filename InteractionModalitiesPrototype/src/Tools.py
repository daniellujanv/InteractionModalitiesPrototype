import cv2
import numpy as np
import matplotlib.pyplot as plt 
import math
''' This module contains sets of functions useful for basic image analysis and should be useful in the SIGB course.
Written and Assembled  (2012,2013) by  Dan Witzner Hansen, IT University.
'''
def showImages(**images):
    plt.figure(1)
    for (counter, (k,v)) in enumerate(images.items()): 
        plt.subplot(1,len(images)  ,counter)
        plt.imshow( cv2.cvtColor(v, cv2.COLOR_BGR2RGB))
        plt.title(k)
        plt.axis('off')
        plt.subplots_adjust(left=0.0, right=1.0, bottom=0.0, wspace=0.1)
    plt.show(block=False)

def getCircleSamples(center=(0,0),radius=1,nPoints=30):
    ''' Samples a circle with center center = (x,y) , radius =1 and in nPoints on the circle.
    Returns an array of a tuple containing the points (x,y) on the circle and the curve gradient in the point (dx,dy)
    Notice the gradient (dx,dy) has unit length'''


    s = np.linspace(0, 2*math.pi, nPoints)
    #points
    P = [(radius*np.cos(t)+center[0], radius*np.sin(t)+center[1],np.cos(t),np.sin(t) ) for t in s ]
    return P



def getImageSequence(fn,fastForward =2):
    '''Load the video sequence (fn) and proceeds, fastForward number of frames.'''
    cap = cv2.VideoCapture(fn)
    for t in range(fastForward):
        running, imgOrig = cap.read()  # Get the first frames
    return cap,imgOrig,running


def getLineCoordinates(p1, p2):
    "Get integer coordinates between p1 and p2 using Bresenhams algorithm"
    " When an image I is given the method also returns the values of I along the line from p1 to p2. p1 and p2 should be within the image I"
    " Usage: coordinates=getLineCoordinates((x1,y1),(x2,y2))"
    
    
    (x1, y1)=p1
    x1=int(x1); y1=int(y1)
    (x2,y2)=p2
    x2 = int(x2);y2=int(y2)
    
    points = []
    issteep = abs(y2-y1) > abs(x2-x1)
    if issteep:
        x1, y1 = y1, x1
        x2, y2 = y2, x2
    rev = False
    if x1 > x2:
        x1, x2 = x2, x1
        y1, y2 = y2, y1
        rev = True
    deltax = x2 - x1
    deltay = abs(y2-y1)
    error = int(deltax / 2)
    y = y1
    ystep = None
    if y1 < y2:
        ystep = 1
    else:
        ystep = -1
    for x in range(x1, x2 + 1):
        if issteep:
            points.append([y, x])
        else:
            points.append([x, y])
        error -= deltay
        if error < 0:
            y += ystep
            error += deltax
    # Reverse the list if the coordinates were reversed
    if rev:
        points.reverse()
       
    retPoints = np.array(points)
    X = retPoints[:,0];
    Y = retPoints[:,1];
    
    
    return retPoints 

class RegionProps:
    '''Class used for getting descriptors of contour-based connected components 
        
        The main method to use is: CalcContourProperties(contour,properties=[]):
        contour: a contours found through cv2.findContours
        properties: list of strings specifying which properties should be calculated and returned
        
        The following properties can be specified:
        
        Area: Area within the contour  - float 
        Boundingbox: Bounding box around contour - 4 tuple (topleft.x,topleft.y,width,height) 
        Length: Length of the contour
        Centroid: The center of contour: (x,y)
        Moments: Dictionary of moments: see 
        Perimiter: Permiter of the contour - equivalent to the length
        Equivdiameter: sqrt(4*Area/pi)
        Extend: Ratio of the area and the area of the bounding box. Expresses how spread out the contour is
        Convexhull: Calculates the convex hull of the contour points
        IsConvex: boolean value specifying if the set of contour points is convex
        
        Returns: Dictionary with key equal to the property name
        
        Example: 
             contours, hierarchy = cv2.findContours(I, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)  
             goodContours = []
             for cnt in contours:
                vals = props.CalcContourProperties(cnt,['Area','Length','Centroid','Extend','ConvexHull'])
                if vals['Area']>100 and vals['Area']<200
                 goodContours.append(cnt)
        '''       
    def __calcArea(self,m,c):
        return cv2.contourArea(c) #,m['m00']
    def __calcLength(self,c):
        return cv2.arcLength(c, True)
    def __calcPerimiter(self,c):
         return cv2.arcLength(c,True)
    def __calcBoundingBox(self,c):
        return cv2.boundingRect(c)
    def __calcCentroid(self,m):
        if(m['m00']!=0):
            retVal =  ( m['m10']/m['m00'],m['m01']/m['m00'] )
        else:   
            retVal = (-1,-1)    
        return retVal
        
    def __calcEquivDiameter(self, m, contur):
        Area = self.__calcArea(m)
        return np.sqrt(4*Area/np.pi)
    def __calcExtend(self,m,c):
        Area = self.__calcArea(m,c)
        BoundingBox = self.__calcBoundingBox(c)
        return Area/(BoundingBox[2]*BoundingBox[3])
    def __calcConvexHull(self,m,c):
        #try:
        CH = cv2.convexHull(c)
        #ConvexArea  = cv2.contourArea(CH)
        #Area =  self.__calcArea(m,c)
        #Solidity = Area/ConvexArea
        return {'ConvexHull':CH} #{'ConvexHull':CH,'ConvexArea':ConvexArea,'Solidity':Solidity}
        #except: 
        #print "stuff:", type(m), type(c)
        
    def CalcContourProperties(self,contour,properties=[]):
        failInInput = False;
        propertyList=[]
        contourProps={};
        for prop in properties:
            prop = str(prop).lower()        
            m = cv2.moments(contour) #Always call moments
            if (prop=='area'):
                contourProps.update({'Area':self.__calcArea(m,contour)}); 
            elif (prop=="boundingbox"):
                contourProps.update({'BoundingBox':self.__calcBoundingBox(contour)});
            elif (prop=="length"):
                contourProps.update({'Length':self.__calcLength(contour)});
            elif (prop=="centroid"):
                contourProps.update({'Centroid':self.__calcCentroid(m)});
            elif (prop=="moments"):
                contourProps.update({'Moments':m});    
            elif (prop=="perimiter"):
                contourProps.update({'Perimiter':self.__calcPerimiter(contour)});
            elif (prop=="equivdiameter"):
                contourProps.update({'EquivDiameter':self.__calcEquiDiameter(m,contour)}); 
            elif (prop=="extend"):
                contourProps.update({'Extend':self.__calcExtend(m,contour)});
            elif (prop=="convexhull"): #Returns the dictionary
                contourProps.update(self.__calcConvexHull(m,contour));  
            elif (prop=="isConvex"):
                    contourProps.update({'IsConvex': cv2.isContourConvex(contour)});
            elif failInInput:   
                    pass   
            else:    
                print "--"*20
                print "*** PROPERTY ERROR "+ prop+" DOES NOT EXIST ***" 
                print "THIS ERROR MESSAGE WILL ONLY BE PRINTED ONCE"
                print "--"*20
                failInInput = True;     
        return contourProps         


class ROISelector:
        
    def __resetPoints(self):
        self.seed_Left_pt = None
        self.seed_Right_pt = None
    
    def __init__(self,inputImg):
        self.img=inputImg.copy()
        self.seed_Left_pt = None
        self.seed_Right_pt = None
        self.winName ='SELECT AN AREA'
        self.help_message = '''This function returns the corners of the selected area as: [(UpperLeftcorner),(LowerRightCorner)]
        Use the Right Button to set Upper left hand corner and and the Left Button to set the lower righthand corner.
        Click on the image to set the area
        Keys:
          Enter/SPACE - OK
          ESC   - exit (Cancel)
        '''
    
    def update(self):
        if (self.seed_Left_pt is None) | (self.seed_Right_pt is None):
            cv2.imshow(self.winName, self.img)
            return
        
        flooded = self.img.copy()
        cv2.rectangle(flooded, self.seed_Left_pt, self.seed_Right_pt,  (0, 0, 255),1)
        cv2.imshow(self.winName, flooded)
    
        
        
    def onmouse(self, event, x, y, flags, param):

        if  flags & cv2.EVENT_FLAG_LBUTTON:
            self.seed_Left_pt = x, y
    #        print seed_Left_pt
    
        if  flags & cv2.EVENT_FLAG_RBUTTON: 
            self.seed_Right_pt = x, y
    #        print seed_Right_pt
        
        self.update()
    def setCorners(self):
        points=[]
    
        UpLeft=(min(self.seed_Left_pt[0],self.seed_Right_pt[0]),min(self.seed_Left_pt[1],self.seed_Right_pt[1]))
        DownRight=(max(self.seed_Left_pt[0],self.seed_Right_pt[0]),max(self.seed_Left_pt[1],self.seed_Right_pt[1]))
        points.append(UpLeft)
        points.append(DownRight)
        return points        
                
    def SelectArea(self,winName='SELECT AN AREA',winPos=(400,400)):# This function returns the corners of the selected area as: [(UpLeftcorner),(DownRightCorner)]
        self.__resetPoints()
        self.winName = winName
        print  self.help_message
        self.update()
        cv2.namedWindow(self.winName, cv2.WINDOW_AUTOSIZE )# cv2.WINDOW_AUTOSIZE
        cv2.setMouseCallback(self.winName, self.onmouse)
        cv2.cv.MoveWindow(self.winName, winPos[0],winPos[1])
        while True:
            ch = cv2.waitKey()

            if ch == 27:#Escape
                cv2.destroyWindow(self.winName)
                return None,False
                break
            if ((ch == 13) or (ch==32)): #enter or space key   
                cv2.destroyWindow(self.winName)    
                return self.setCorners(),True
                break


def rotateImage(I, angle):
    "Rotate the image, I, angle degrees around the image center"
    size = I.shape
    image_center = tuple(np.array(size)/2)
    rot_mat = cv2.getRotationMatrix2D(image_center[0:2],angle,1)
    result = cv2.warpAffine(I, rot_mat,dsize=size[0:2],flags=cv2.INTER_LINEAR)
    return result
