# import the necessary packages
import numpy as np
import cv2


def DrawCircle():   
    output = cv2.imread('checkstamp_rectangle.jpg')
    gray = cv2.cvtColor(output, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray,(5,5),0);
    gray = cv2.medianBlur(gray,5)
    
    # Adaptive Guassian Threshold is to detect sharp edges in the Image. For more information Google it.
    gray = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,11,3.5)
    kernel = np.ones((2,2),np.uint8)
    gray = cv2.erode(gray,kernel,iterations = 1)    
    gray = cv2.dilate(gray,kernel,iterations = 1)
    
    # cv2.imshow('test', gray)
    # cv2.waitKey(0)

    circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1.2, 100)
    print(circles)
    #ensure at least some circles were found
    if circles is not None:
        # convert the (x, y) coordinates and radius of the circles to integers
        circles = np.round(circles[0, :]).astype("int")      
        # loop over the (x, y) coordinates and radius of the circles
        for (x, y, r) in circles:
            # draw the circle in the output image, then draw a rectangle in the image
            # corresponding to the center of the circle
            cv2.circle(output, (x, y), r, (0, 255, 0), 4)            
           
        cv2.imwrite('checkstamp_circle.jpg', output)

def SharpenImage(img): 
    #Create the identity filter, but with the 1 shifted to the right!
    kernel = np.zeros( (100,100), np.float32)
    kernel[2,2] = 2.0   #Identity, times two! 

    #Create a box filter:
    boxFilter = np.ones( (100,100), np.float32) / 10000.0

    #Subtract the two:
    kernel = kernel - boxFilter

    #Note that we are subject to overflow and underflow here...but I believe that
    # filter2D clips top and bottom ranges on the output, plus you'd need a
    # very bright or very dark pixel surrounded by the opposite type.

    custom = cv2.filter2D(img, -1, kernel)

    return custom
    

#Load Image 
img = cv2.imread('checkstamp.jpg')
sharpimgage = SharpenImage(img)
imgray = cv2.cvtColor(sharpimgage,cv2.COLOR_BGR2GRAY)
gray = cv2.GaussianBlur(imgray, (3, 3), 0)

# detect edges in the image
edged = cv2.Canny(imgray, 100, 300)

# construct and apply a closing kernel to 'close' gaps between 'white'
# pixels
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
closed = cv2.morphologyEx(edged, cv2.MORPH_CLOSE, kernel)

# find contours (i.e. the 'outlines') in the image and initialize the
# total number of books found
im2, cnts, h = cv2.findContours(closed.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
total = 0

# loop over the contours
for c in cnts:
    # approximate the contour
    # draw rectangle 
    peri = cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, 0.02 * peri, True)

    # if the approximated contour has four points, then assume that the
    # contour is a book -- a book is a rectangle and thus has four vertices
    if len(approx) == 4:
        cv2.drawContours(sharpimgage, [approx], -1, (0, 255, 0), 4)
        total += 1 

cv2.imwrite('checkstamp_rectangle.jpg', sharpimgage)

# DrawCircle()