import numpy as np
import cv2


# reading fiel img:
F = cv2.imread('../2D_field.png')

# 4 points in the 2D_field.png
points1 = np.array([(156, 151),
                    (525, 0),
                    (525, 700),
                    (1050-156, 151)]).astype(np.float32)
# 4 points in the output.mp4
points2 = np.array([(156, 168),
                    (686, 107),
                    (926, 792),
                    (1202, 117)]).astype(np.float32)

# compute homography from point correspondences
H = cv2.getPerspectiveTransform(points2, points1)
output_size = (F.shape[1], F.shape[0])

# craeting background subtractor:
# backSub = cv2.createBackgroundSubtractorMOG2(100, 100, True)
backSub = cv2.createBackgroundSubtractorKNN(100, 1000, True) # history, treshold, detect_shadow


def preprocess(img):
    # apply BGS:
    fgMask = backSub.apply(img) # [0, 127, 255]
    fgMask = np.uint8(fgMask > 127) * 255
    
    # # closing
    kernel = np.ones((3,3), np.uint8)
    C = cv2.morphologyEx(fgMask, cv2.MORPH_CLOSE, kernel)

    kernel = np.ones((5,3), np.uint8)
    D1 = cv2.dilate(C, kernel)
    
    kernel = np.ones((2,3), np.uint8)
    E = cv2.erode(D1, kernel)

    kernel = np.ones((7,5), np.uint8)
    D = cv2.dilate(E, kernel)
    return D


def CP_analysis(img):
    # connected components with statistics
    n, _, stats, _ = cv2.connectedComponentsWithStats(img)
    return(n, stats)


def get_circle_centers(n, stats):
    foots = list()
    for i in range(1, n):
        alpha = stats[i][1] / output_size[1]
        # if stats[i][3] > 120 - 90*alpha: # height of CP 
        if stats[i][4] > 600 - 300*(alpha**2): # area of CP
            x = stats[i][0] + stats[i][2]//2
            y = stats[i][1] + int(1*stats[i][3])
            foots.append((x, y))
    return foots