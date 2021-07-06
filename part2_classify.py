# computer vision - 5 July 2021 - final project

import numpy as np
import cv2
import time
import torch


import utils
from train_nn import Net, transform, device


net = Net().double().to(device)
net.load_state_dict(torch.load("./model2021-07-06 19:23:16.549593.idk"))
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
    foots = []
    for i in range(1, n):
        alpha = stats[i][1] / utils.output_size[1]
        if stats[i][4] > 500 - 300*(alpha**2): # area of CP
            ppatch = proj_img[stats[i][1]:stats[i][1]+stats[i][3] ,stats[i][0]:stats[i][0]+stats[i][2]]
            cv2.imshow("ppatch", ppatch)
            ppatch = cv2.resize(ppatch, (50, 100))
            ppatch = np.float64(ppatch)/255.0
            ppatch = transform(ppatch)
            ppatch = ppatch.reshape(1, 100, 50, 3)
            ppatch = ppatch.permute(0, 3, 1, 2).double().to(device)
            output = net(ppatch)
            print(output)
            x = stats[i][0] + stats[i][2]//2
            y = stats[i][1] + stats[i][3]
            if output > 0.7:
                color = [255, 0, 0]
                foots.append({'pos': (x, y), 'color': color})

            elif output < 0.3:
                color = [0, 0, 255]
                foots.append({'pos': (x, y), 'color': color})
            else:
                color = [0, 255, 0]
                foots.append({'pos': (x, y), 'color': color})
        
    # draw circles on foot steps:
    F_circle = np.array(utils.F)
    for f in foots:
        cv2.circle(F_circle, f['pos'], 3, f['color'], 5)
        cv2.circle(proj_img, f['pos'], 3, f['color'], 5)
        
    
    # Display some images
    cv2.imshow('win1', I)
    cv2.imshow('win2', proj_img)
    cv2.imshow('dilate(E)', preproc_img)
    cv2.imshow('2D_field', F_circle[::2, ::2])

    key = cv2.waitKey(1) 

    if key & 0xFF == ord('q'): 
        break


print("#frames: " + str(num_frames) + " elapsed time: " + str(time.time() - t))
cap.release()
cv2.destroyAllWindows()
torch.cuda.empty_cache()

