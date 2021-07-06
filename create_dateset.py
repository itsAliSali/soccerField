import glob
import os
import random


train_test_ratio = 0.8


# copying file into ./data/dataset/
players = glob.glob("./data/humanCntrl/all/*.jpg")

random.shuffle(players)

for i in range(0, round(train_test_ratio*len(players))):
    p = players[i]
    os.system(f'cp {p} ./data/dataset/train')

for i in range(round(train_test_ratio*len(players)), len(players)):
    p = players[i]
    os.system(f'cp {p} ./data/dataset/test/')


# renaming files --> idx_0/1.jpg. (r: 0, b: 1)
train = glob.glob("./data/dataset/train/*.jpg")
idx = 1
for t in train:
    if t[-6:] == "_r.jpg":
        os.system(f'mv {t} ./data/dataset/train/{idx}_0.jpg')
    elif t[-6:] == "_b.jpg":
        os.system(f'mv {t} ./data/dataset/train/{idx}_1.jpg')
    idx += 1    

test = glob.glob("./data/dataset/test/*.jpg")
idx = 1
for t in test:
    if t[-6:] == "_r.jpg":
        os.system(f'mv {t} ./data/dataset/test/{idx}_0.jpg')
    elif t[-6:] == "_b.jpg":
        os.system(f'mv {t} ./data/dataset/test/{idx}_1.jpg')
    idx += 1    