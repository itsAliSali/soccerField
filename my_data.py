import os
import torch
from torch.utils.data import Dataset, DataLoader
import cv2
import numpy as np
import glob

class SoccerFieldDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform

        self.imgs_path = glob.glob(self.root_dir + "*.jpg")

    def __len__(self):
        return len(self.imgs_path)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name_r = os.path.join(self.root_dir, f'{idx}_0.jpg')
        img_name_b = os.path.join(self.root_dir, f'{idx}_1.jpg')
        
        if os.path.isfile(img_name_r):
            img_path = img_name_r
            label = 0
        else:
            img_path = img_name_b
            label = 1

        image = cv2.imread(img_path)
        image = cv2.resize(image, (50, 100))

        sample = {'img': image, 'tag': label}

        if self.transform:
            sample = self.transform(sample)

        return sample



if __name__ == "__main__":
    
    ds = SoccerFieldDataset('./data/dataset/train/')

    print(ds.__len__())

    d0 = ds.__getitem__(0)
    d1 = ds.__getitem__(175)
    d2 = ds.__getitem__(187)

    cv2.imshow('0', d0['img'])
    cv2.imshow('175', d1['img'])
    cv2.imshow('187', d2['img'])

    cv2.waitKey(0)
    cv2.destroyAllWindows()    