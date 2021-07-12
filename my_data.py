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
        image = np.float64(image/255)

        if self.transform:
            image = self.transform(image)

        return image, label



if __name__ == "__main__":
    
    import random 
    import matplotlib.pyplot as plt

    from train_nn import Net, transform, device


    net = Net().double().to(device)
    net.load_state_dict(torch.load("./model2021-07-11 13:05:41.314871.idk"))
    net.eval()

    ds = SoccerFieldDataset('./data/dataset/test/', transform)

    print("dataset len:", ds.__len__())

    w, h = 10, 4
    f, ax = plt.subplots(h, w)

    for i in range(w*h):
        idx = random.randint(0, ds.__len__())
        img, lbl = ds.__getitem__(idx)
        imgo = img.cpu().permute(1, 2, 0).numpy()
        
        img = img.reshape(1, 3, 100, 50).double().to(device)
    
        output = net(img)
    
        ax[i//w,i%w].imshow(imgo)
        ax[i//w,i%w].title.set_text(f'{lbl}/{float(output):.4f}')
        ax[i//w,i%w].axis('off')
    
    plt.show()
    torch.cuda.empty_cache()
