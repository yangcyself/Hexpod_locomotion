import torch
import numpy as np
import torchvision.datasets as dset
import torch.nn as nn
import torchvision.transforms as transforms
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, sampler
import pickle as pkl
# import math
from math import cos, sin,pi


class FCnet(nn.Module):
    def __init__(self):
        super(FCnet, self).__init__()
        putsize = 3
        hidden_dim1 = 10
        hidden_dim2 = 10
        self.fc1 = nn.Linear(putsize,hidden_dim1)
        self.fc2 = nn.Linear(hidden_dim1,hidden_dim2)
        self.fc3  =nn.Linear(hidden_dim2,putsize)
        self.tanh = nn.Tanh()
        
    def forward(self,x):
        hidden1 = self.tanh(self.fc1(x))
        hidden2 = self.tanh(self.fc2(hidden1))
        out = self.tanh(self.fc3(hidden2))
        out = out/1.3 + 0.02
        return out

class MyDataset(torch.utils.data.Dataset):
    
    def create_dataset(self, datasets):
        dataX = []
        dataY = []
        for dataset in datasets:
            # dataset = np.array(dataset)
            # dataset.astype(np.float32)
            dataX +=dataset[0]
            dataY +=dataset[1]
        return np.array(dataX), np.array(dataY)


    def __init__(self, dataset):
        super().__init__()
        self.dataset = dataset
        self.X, self.Y = self.create_dataset(self.dataset)
        print(self.X.shape)
        print(self.Y.shape)

        self.X = torch.from_numpy(self.X)
        self.Y = torch.from_numpy(self.Y)
    

    def __getitem__(self, index):
        x = self.X[index]
        y = self.Y[index]
        return x, y

    def __len__(self):
        return self.X.shape[0]

class footControler:
    def __init__(self,modelfile):
        self.model = FCnet()
        ckpt = torch.load(modelfile)
        self.model.load_state_dict(ckpt)
        # self.angletable = np.array([1,2,3,4,5,6])
        # self.angletable = self.angletable * math.pi / 6
        self.trans = []
        for i in range(6):
            angle = i*pi/3
            self.trans.append(np.array([[cos(angle),-sin(angle),0],[sin(angle),cos(angle),0],[0,0,1]]))
        self.trans[3],self.trans[5] = self.trans[5],self.trans[3]
    def position(self,pos,no):
        pos = np.array(pos).dot(self.trans[no])
        print (pos)
        return self.model(torch.from_numpy(pos).float()).detach().numpy()
    def posture(self,poss):
        res = []
        for i,p in enumerate(poss):
            res.append(self.position(p,i))
        return res


        


if __name__ == "__main__":

    use_cuda = torch.cuda.is_available()
    # use_cuda = False
    device = torch.device("cuda" if use_cuda else "cpu")
    fc = FCnet().to(device)
    optimizer = optim.SGD(fc.parameters(), lr=0.001, momentum=0.2)#weight_decay=0.00005)
    batch_size = 50
    # train
    train_epoch = 400
    
    with open("data.pkl","rb") as f:
        dt = pkl.load(f)
    with open("data2.pkl","rb") as f:
        dt1 = pkl.load(f)
    with open("data3.pkl","rb") as f:
        dt2 = pkl.load(f)
    # with open("data4.pkl","rb") as f:
    #     dt3 = pkl.load(f)
    loss_F = torch.nn.MSELoss()
    # setleglength(clientID,pole,dt[0][1])
    train_dataset = MyDataset([dt,dt1,dt2])
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size, 
        shuffle=True,
    )
    
    for epoch in range(train_epoch): 
        fc.train()
        print("epoch:", epoch)
        for step, input_data in enumerate(train_dataloader):
            # print (input_data)
            # break
            # x, y = input_data
            y, x = input_data
            x = x.to(device).float()
            y = y.to(device).float()
            
            pred_y = fc(x)

            loss = loss_F(pred_y, y) # 计算loss
            
            if step %50 == 49: # 每50步，计算精度
                
                print("{}/{} steps".format(step, len(train_dataloader)), float(loss))
            # print(x[:2], pred_y[:2])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    print("SAVED! ", 'firstshot4.pt')
    torch.save(fc.state_dict(), 'firstshot4.pt')


