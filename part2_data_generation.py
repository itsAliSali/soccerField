# computer vision - 5 July 2021 - final project

import numpy as np
import cv2
import time

import utils

# create a VideoCapture object
cap = cv2.VideoCapture('../output.mp4')
fps = cap.get(cv2.CAP_PROP_FPS)
mspf = round(1000/fps)

num_frames = 0
t = time.time()
while True:
    
    # Capture frame-by-frame
    ret, I = cap.read()
    
    if ret == False: # end of video (perhaps)
        break
    num_frames += 1

    # project the video to the field coordinate.
    proj_img = cv2.warpPerspective(I, utils.H, utils.output_size)
    
    # some closing, openning, dilating, ...
    preproc_img = utils.preprocess(proj_img)

    # apply connected components Alg and find foot position:
    n, stats = utils.CP_analysis(preproc_img)

    # save each player (patch) + label
    if (num_frames) % 20 == 0:
        for i in range(1, n):
            alpha = stats[i][1] / utils.output_size[1]
            if stats[i][4] > 500 - 300*(alpha**2): # area of CP
                ppatch = proj_img[stats[i][1]:stats[i][1]+stats[i][3] ,stats[i][0]:stats[i][0]+stats[i][2]]
                print(cv2.imwrite(f'./data/{num_frames}_{i}.jpg', ppatch)       )
            
    # Display some images
    cv2.imshow('win1', I)
    cv2.imshow('win2', proj_img)
    cv2.imshow('dilate(E)', preproc_img)
    # cv2.imshow('2D_field', F_circle[::2, ::2])

    key = cv2.waitKey(mspf//15+3) 

    if key & 0xFF == ord('q'): 
        break


print("#frames: " + str(num_frames) + " elapsed time: " + str(time.time() - t))
cap.release()
cv2.destroyAllWindows()


