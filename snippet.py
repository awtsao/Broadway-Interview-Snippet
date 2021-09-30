import cv2
import itertools
import numpy as np
import time
from imutils.video import WebcamVideoStream
def getTwoLargest(contours):
    '''
    This function returns the indices of two largest contours.
    :param contours: List of cv2 contour objects
    :return: Index of the contours with the two largest area
    '''
    x = cv2.contourArea(contours[0])
    y = cv2.contourArea(contours[1])

    largest = 0 if x > y else 1
    largest_area = x if x > y else y
    second = 1 if x > y else 0
    second_area = y if x > y else x

    for i in range(2, len(contours)):
        area = cv2.contourArea(contours[i])
        if area > largest_area:
            # The old largest area is now the 2nd largest area
            second_area = largest_area
            # Set the largest area to the new area
            largest_area = area
            # Set 2nd largest index to old largest index
            second = largest
            # Set largest index to i
            largest = i
        elif area > second_area:
            # New area is in between old 2nd largest and largest area so new area becomes 2nd largest area.
            second_area = area
            second = i

    if second_area < 4000:
        # We need the area to be larger than some pre-defined value to count as a "hand". We arbitrarily chose 4000
        second = -1
    return largest, second


def getHands(contours):
    '''
    This function returns the contours of the two largest tuple objects which should correspond to the hands
    in the frame.
    :param contours: List of cv2 contour objects
    :return: A tuple containing the two largest tuple objects, if they exist. left
    is either None or the left contour object/hand. Right is always the right contour object/ahnd.
    '''
    if len(contours) < 2:
        return None, contours[0]
    # Grab the two largest contours
    largest, second = getTwoLargest(contours)
    if second < 0:
        # No left hand so return immediately
        return None, contours[largest]

    first_m = cv2.moments(contours[largest])
    first_x = int(first_m["m10"] / first_m["m00"])
    second_m = cv2.moments(contours[second])
    second_x = int(second_m["m10"] / second_m["m00"])
    left = contours[largest] if first_x < second_x else contours[second]
    right = contours[second] if first_x < second_x else contours[largest]
    return left, right


def getFingerTip(defects, contour, centroid, h):
    '''
    This function finds the farthest defect in the input contour and checks if it
    should be detected as a fingertip. If so, it returns True and the location of the defect/fingertip.
    Otherwise, is returns False and None.
    :param defects: list of defects of input contour
    :param contour: right hand contour
    :param centroid: centroid of right hand contour
    :return: -detected: boolean corresponding to whether fingertip is detected or not
             -farthest: location of farthest defect
    '''
    if defects is not None and centroid is not None:
        cx, cy = centroid

        # Get start points of all defects
        s = defects[:, 0][:, 0]

        # Get x and y coordinates of all defects
        sx = np.array(contour[s][:, 0][:, 0], dtype=np.float)
        sy = np.array(contour[s][:, 0][:, 1], dtype=np.float)

        # Calculate distance from centroid of contour to each defect
        x_dist = (cx - sx) ** 2
        y_dist = (cy - sy) ** 2
        dist = np.sqrt(y_dist + x_dist)

        # This one grabs all the indices of the defects below the centroid
        indices = np.nonzero((cy - sy) < 0)[0]

        # set all points found below the centroid to negative distance.
        dist[indices] = -1
        # Grab the defect with the largest distance that is above the centroid.
        highest_index = np.argmax(dist)

        # Calculate the ratio of the distance of the defect and centroid
        yratio = (h - sy[highest_index]) / (h - cy)

        if highest_index < len(s) and yratio > 2:
            # Get index of farthest defect
            farthest_s = s[highest_index]
            # Grab coordinate of fingertip
            farthest = tuple(contour[farthest_s][0])
            return True, farthest
        else:
            return False, None
    return False, None