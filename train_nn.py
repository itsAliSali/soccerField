import torch
import torchvision
import torchvision.transforms as transforms

import my_data as ds

import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# print(">>>>> ", device)


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(3168, 520)
        self.fc2 = nn.Linear(520, 40)
        self.fc3 = nn.Linear(40, 1)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x))).to(device)
        x = self.pool(F.relu(self.conv2(x))).to(device)
        x = torch.flatten(x, 1).to(device) # flatten all dimensions except batch
        x = F.relu(self.fc1(x)).to(device)
        x = torch.sigmoid(self.fc2(x)).to(device)
        x = torch.sigmoid(self.fc3(x).to(device))
        return x


transform = transforms.Compose([transforms.ToTensor()])#, transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])


if __name__ == "__main__":
   

    batch_size = 10

    trainset = ds.SoccerFieldDataset('./data/dataset/train/', transform=transform)

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                            shuffle=True, num_workers=2)

    testset = ds.SoccerFieldDataset('./data/dataset/test/', transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=1,
                                            shuffle=False, num_workers=2)


    net = Net()
    net = net.double()
    net.to(device)

    import torch.optim as optim

    # criterion = nn.CrossEntropyLoss()
    criterion = nn.MSELoss()
    optimizer = optim.SGD(net.parameters(), lr=0.1, momentum=0.05)


    for epoch in range(10):  # loop over the dataset multiple times

        # training
        loss_train = 0.0
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data
            # inputs = inputs.permute(0, 3, 1, 2).to(device)
            inputs = inputs.to(device)
            labels = labels.reshape((-1,1)).double().to(device)
        
            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            loss_train += loss.item()
            if i % 100 == 99:    # print every 100 mini-batches
                print('[%d, %5d] loss: %.6f' %
                    (epoch + 1, i + 1, running_loss / 100 ))
                running_loss = 0.0

        print('>>>> epoch: %d, train_loss: %.6f' %
                    (epoch + 1, loss_train / trainloader.__len__() ))
        

        # test
        loss_test = 0 
        num = 0
        for i, data in enumerate(testloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data
            # print(inputs.shape)
            # inputs = inputs.permute(0, 3, 1, 2).to(device)
            inputs = inputs.to(device)
            labels = labels.reshape((-1,1)).double().to(device)
            
            # forward 
            outputs = net(inputs)
            if outputs > 0.5 and labels == 1:
                num += 1
            if outputs < 0.5 and labels == 0:
                num += 1
            
            loss = criterion(outputs, labels)
            loss_test += loss.item()
        
        print('>>>> epoch: %d, test _loss: %.6f' %
                    (epoch + 1, loss_test / testloader.__len__() ))
        print(f"acc= {num/testloader.__len__() *100 }")
        

    from datetime import datetime as dt
    torch.save(net.state_dict(), f"./model{str(dt.now())}.idk")

    print('Finished Training')
    torch.cuda.empty_cache()


    # import random 
    # import matplotlib.pyplot as plt
    # import cv2
    # import numpy as np
    # dss = ds.SoccerFieldDataset('./data/dataset/train/', transform)

    # print(dss.__len__())

    # # d0, l0 = ds.__getitem__(0)
    # # d1, l1 = ds.__getitem__(175)
    # # d2, l2 = ds.__getitem__(187)

    # f, ax = plt.subplots(4,5)

    # for i in range(20):
    #     idx = random.randint(0, dss.__len__())
    #     img, lbl = dss.__getitem__(idx)
    #     imgo = img.cpu().permute(1,2,0).numpy()
    #     ##
    #     # img = cv2.resize(img, (50, 100))
    #     # img = np.float64(img/255)
    #     # img = transform(img)
    #     img = img.reshape(1, 3, 100, 50).double().to(device)
    #     # img = img.permute(0, 3, 1, 2).double().to(device)
    #     output = net(img)
    # ##
    #     ax[i//5,i%5].imshow(imgo)
    #     ax[i//5,i%5].title.set_text(f'{lbl}/{float(output)}')
    #     ax[i//5,i%5].axis('off')
    
    # plt.show()
    # torch.cuda.empty_cache()