import cv2
import random
import sys
import imutils
import numpy as np
from imutils.object_detection import non_max_suppression
from matplotlib import pyplot as plt


VID_DIR = "/Users/drew/Documents/Python/tracking/images/"
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
    
def contours(threshed_img, img):
    contours, hier = cv2.findContours(threshed_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for c in contours:
        # get the bounding rect
        x, y, w, h = cv2.boundingRect(c)
        # draw a green rectangle to visualize the bounding rect
        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
        # get the min area rect
        rect = cv2.minAreaRect(c)
        box = cv2.boxPoints(rect)
        # convert all coordinates floating point values to int
        box = np.int0(box)
        # draw a red 'nghien' rectangle
        cv2.drawContours(img, [box], 0, (0, 0, 255))
    cv2.drawContours(img, contours, -1, (255, 255, 0), 1)
    return img


def get_rects(img):
    display(img, "before")
    rects, weights = hog.detectMultiScale(img, winStride=(4, 4), padding=(8, 8), scale=1.05)

    for (x, y, w, h) in rects:
        #cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 0)
        rects = np.array([[x, y, x + w, y + h] for (x, y, w, h) in rects])

    result = non_max_suppression(rects, probs=None, overlapThresh=0.7)
    print("#people: ", len(result))
    for (xA, yA, xB, yB) in result:
        clr = (random.randint(0,255),random.randint(0,255),random.randint(0,255))
        cv2.rectangle(img, (xA, yA), (xB, yB), clr, 3)
    display(img, "after")
    return img

def display(img, title):
    cv2.imshow(title, img)
    key = cv2.waitKey(3000)
    if key == 27:
        cv2.destroyAllWindows()
        
def main():
    filename = VID_DIR + str(sys.argv[1])
    img = cv2.pyrDown(cv2.imread(filename, cv2.IMREAD_UNCHANGED))
    display(img, "img")
    
    ret, threshed_img = cv2.threshold(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), 127, 255, cv2.THRESH_BINARY)
    display(threshed_img, "thresh")

    contours_img = contours(threshed_img, img.copy())
    display(contours_img, "contours")
    
    rects_img = get_rects(img.copy())
    #display(rects_img, "rects")
    """
    plt.imshow(img)
    plt.show(block=False)
    plt.pause(3)
    plt.close()
    """
if __name__ == '__main__':
    main()
    cv2.destroyAllWindows()
