# computer vision - 5 July 2021 - final project

import numpy as np
import cv2
import time
import torch


import utils
from train_nn import Net


net = Net()
net.load_state_dict(torch.load("./model2021-07-06 17:44:22.614611.idk"))
net.eval()

# create a VideoCapture object
cap = cv2.VideoCapture('../output.mp4')
fps = cap.get(cv2.CAP_PROP_FPS)
mspf = round(1000/fps)

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
    proj_img = cv2.warpPerspective(I, utils.H, utils.output_size)
    
    # some closing, openning, dilating, ...
    preproc_img = utils.preprocess(proj_img)

    # apply connected components Alg and find foot position:
    n, stats = utils.CP_analysis(preproc_img)
    foots = utils.get_circle_centers(n, stats)

    # draw circles on foot steps:
    F_circle = np.array(utils.F)
    for f in foots:
        cv2.circle(F_circle, f, 3, [0,0,255], 5)
        cv2.circle(proj_img, f, 3, [0,0,255], 5)
        
    
    # Display some images
    cv2.imshow('win1', I)
    cv2.imshow('win2', proj_img)
    cv2.imshow('dilate(E)', preproc_img)
    cv2.imshow('2D_field', F_circle[::2, ::2])

    key = cv2.waitKey(mspf//15+3) 

    if key & 0xFF == ord('q'): 
        break


print("#frames: " + str(num_frames) + " elapsed time: " + str(time.time() - t))
cap.release()
cv2.destroyAllWindows()


