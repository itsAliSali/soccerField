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
backSub = cv2.createBackgroundSubtractorKNN(100, 1000, True)

num_frames = 0
t = time.time()
while True:
    
    # Capture frame-by-frame
    ret, I = cap.read()
    
    if ret == False: # end of video (perhaps)
        break
    num_frames += 1

    # project the video to the field coordinate.
    J = cv2.warpPerspective(I, H, output_size)
    
    # apply BGS:
    fgMask = backSub.apply(J)
    
    # Display I, J
    # cv2.imshow('win1', I)
    # cv2.imshow('win2', J)
    cv2.imshow('win3', fgMask)
    

    key = cv2.waitKey(mspf//10) 

    if key & 0xFF == ord('q'): 
        break

print("#frames: " + str(num_frames) + " elapsed time: " + str(time.time() - t))
cap.release()
cv2.destroyAllWindows()


