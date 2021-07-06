import os
import torch
from torch.utils.data import Dataset
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
        image = np.float64(image /255)
        # image = torch.from_numpy(image).long()
        
        # sample = {'image': image, 'tag': label}

        if self.transform:
            sample = self.transform(image)

        return image, label



if __name__ == "__main__":
    
    import random 
    import matplotlib.pyplot as plt

    ds = SoccerFieldDataset('./data/dataset/train/')

    print(ds.__len__())

    # d0, l0 = ds.__getitem__(0)
    # d1, l1 = ds.__getitem__(175)
    # d2, l2 = ds.__getitem__(187)

    f, ax = plt.subplots(4,5)

    for i in range(20):
        idx = random.randint(0, ds.__len__())
        img, lbl = ds.__getitem__(idx)
        ax[i//5,i%5].imshow(img)
        ax[i//5,i%5].title.set_text(f'{lbl}')
        ax[i//5,i%5].axis('off')
    
    plt.show()
