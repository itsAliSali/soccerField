# computer vision - 5 July 2021 - final project

import numpy as np
import cv2
import time


# reading fiel img:
F = cv2.imread('../2D_field.png')
# cv2.imshow('field', F)

# create a VideoCapture object
cap = cv2.VideoCapture('../output.mp4')
fps = cap.get(cv2.CAP_PROP_FPS)
mspf = round(1000/fps)
# w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))  # width of the frame
# h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) # height of the frame

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

# kernel used for closing:
kernel = np.ones((20,2),np.uint8)

num_frames = 0
t = time.time()
while True:
    
    # Capture frame-by-frame
    ret, I = cap.read()
    # I = cv2.GaussianBlur(I, (5, 3), 0)

    if ret == False: # end of video (perhaps)
        break
    num_frames += 1

    # project the video to the field coordinate.
    J = cv2.warpPerspective(I, H, output_size)
    # J = cv2.GaussianBlur(J, (3, 1), 0)
    # apply BGS:
    fgMask = backSub.apply(J) # [0, 127, 255]
    fgMask = np.uint8(fgMask > 127) * 255
    
    # # closing
    kernel = np.ones((3,3), np.uint8)
    C = cv2.morphologyEx(fgMask, cv2.MORPH_CLOSE, kernel)

    kernel = np.ones((5,3), np.uint8)
    D1 = cv2.dilate(C, kernel)
    
    kernel = np.ones((1,3), np.uint8)
    E = cv2.erode(D1, kernel)

    kernel = np.ones((7,4), np.uint8)
    D = cv2.dilate(E, kernel)

    # connected components with statistics
    n, _, stats, _ = cv2.connectedComponentsWithStats(D)

    # Top = np.zeros(output_size, dtype=np.uint8)
    foots = list()
    for i in range(1, n):
        alpha = stats[i][1] / output_size[1]
        if stats[i][3] > 90 - 60*alpha: # height of CP 
            x = stats[i][0] + stats[i][2]//2
            y = stats[i][1] + int(1*stats[i][3])
            foots.append((x, y))
    
    F_circle = np.array(F)
    for f in foots:
        cv2.circle(F_circle, f, 3, [0,0,255], 5)
        cv2.circle(J, f, 3, [0,0,255], 5)
        
    
    # Display I, J
    cv2.imshow('win1', I)
    cv2.imshow('win2', J)
    cv2.imshow('fgmask', fgMask)
    cv2.imshow('Closing(fg)', C)
    # cv2.imshow('openning(fg)', O)
    cv2.imshow('erode(C)', E)
    cv2.imshow('dilate(E)', D)
    cv2.imshow('2D_field', F_circle)


    key = cv2.waitKey(mspf//10) 

    if key & 0xFF == ord('q'): 
        break

print("#frames: " + str(num_frames) + " elapsed time: " + str(time.time() - t))
cap.release()
cv2.destroyAllWindows()


