import torch
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import my_data as ds

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 8, 5)
        self.fc1 = nn.Linear(8 * 9 * 22, 520)
        self.fc2 = nn.Linear(520, 40)
        self.fc3 = nn.Linear(40, 1)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x))).to(device)
        x = self.pool(F.relu(self.conv2(x))).to(device)
        x = torch.flatten(x, 1).to(device) # flatten all dimensions except batch
        x = F.relu(self.fc1(x)).to(device)
        x = F.relu(self.fc2(x)).to(device)
        x = torch.sigmoid(self.fc3(x).to(device))
        return x


transform = transforms.Compose([transforms.ToTensor()])#, transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])


if __name__ == "__main__":   

    batch_size = 10

    trainset = ds.SoccerFieldDataset('./data/dataset/train/', transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                            shuffle=True, num_workers=2)

    testset = ds.SoccerFieldDataset('./data/dataset/test/', transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                            shuffle=False, num_workers=2)

    net = Net()
    net = net.double()
    net.to(device)

    criterion = nn.MSELoss()
    optimizer = optim.SGD(net.parameters(), lr=0.1, momentum=0.05)

    for epoch in range(10): 
        print("#"*10, f'epoch {epoch+1}', "#"*10)
        
        # training
        net.train()
        loss_train = 0.0
        running_loss = 0.0
        num = 0
        for i, data in enumerate(trainloader, 0):
            
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.reshape((-1,1)).double().to(device)
        
            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # accuracy
            corrects = (outputs > 0.5) == labels
            num += sum(corrects)
            
            # print statistics
            running_loss += loss.item()
            loss_train += loss.item()
            if i % 100 == 99:    # print every 100 mini-batches
                print('[%d, %5d] loss: %.6f' %
                    (epoch + 1, i + 1, running_loss / 100 ))
                running_loss = 0.0

        # print('>>>> epoch: %d, train_loss: %.6f' %
        #             (epoch + 1, loss_train / trainset.__len__() ))
        print(f"train_acc = {float(num/trainset.__len__() * 100):.4f}")

        # test
        net.eval()
        loss_test = 0 
        num = 0
        for i, data in enumerate(testloader, 0):
            
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.reshape((-1,1)).double().to(device)
            
            # forward 
            outputs = net(inputs)
            
            # accuracy
            corrects = (outputs > 0.5) == labels
            num += sum(corrects)
            
            loss = criterion(outputs, labels)
            loss_test += loss.item()
        
        print(f"test_acc = {float(num/testset.__len__() * 100):.4f}\n")
        

    from datetime import datetime as dt
    torch.save(net.state_dict(), f"./model{str(dt.now())}.idk")

    print('Finished Training')
    torch.cuda.empty_cache()
