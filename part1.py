# computer vision - 5 July 2021 - final project

import numpy as np
import cv2
import time
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
    # Display I
    cv2.imshow('win1',I)
    
    key = cv2.waitKey(mspf) 

    if key & 0xFF == ord('q'): 
        break

print("#frames: " + str(num_frames) + " elapsed time: " + str(time.time() - t))
cap.release()
cv2.destroyAllWindows()


