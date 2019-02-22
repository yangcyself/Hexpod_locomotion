# -^- coding:utf-8 -^-
"""
The low level controller of the system
When used as a module: querry and give feedback
When used as main: learn in an supervised learning approach
"""
# -^- coding:utf-8 -^-
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
import sys
sys.path.append("../")
from actorcritic.logger import Logger


class FCnet(nn.Module):
    def __init__(self):
        super(FCnet, self).__init__()
        inputsize = 6
        outputsize = 3
        hidden_dim1 = 128
        hidden_dim2 = 128
        self.fc1 = nn.Linear(inputsize,hidden_dim1)
        self.fc2 = nn.Linear(hidden_dim1,hidden_dim2)
        self.fc3  =nn.Linear(hidden_dim2,outputsize)
        self.tanh = nn.Tanh()
        
    def forward(self,x):
        hidden1 = self.tanh(self.fc1(x))
        hidden2 = self.tanh(self.fc2(hidden1))
        out = self.tanh(self.fc3(hidden2))
        out = out  #[-0.05,0.05] 精度能保证到小数点后面三个0
        return out

class MyDataset(torch.utils.data.Dataset):
    
    def create_dataset(self, datasets):
        dataX = []
        dataY = []
        for dataset in datasets:
            # dataset = np.array(dataset)
            # dataset.astype(np.float32)

            for l,p in zip(dataset["ls"],dataset["dps"]):
                # print(type(l))
                # print(np.concatenate((l,p)))
                dataX.append(np.concatenate((l,p)))

            dataY +=dataset["dls"]
        return np.array(dataX), np.array(dataY)


    def __init__(self, dataset):
        super().__init__()
        self.dataset = dataset
        self.X, self.Y = self.create_dataset(self.dataset)
        # print(self.X.shape)
        # print(self.Y.shape)

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
        basAng = [1,3,5,-1,-3,-5]
        basAng = (np.array(basAng)-1)*pi/6
        for angle in basAng:
            # angle = i*pi/3
            self.trans.append(np.array([[1,0,0,0,0,0],[0,1,0,0,0,0],[0,0,1,0,0,0],[0,0,0,cos(angle),-sin(angle),0],[0,0,0,sin(angle),cos(angle),0],[0,0,0,0,0,1]]))
        # self.trans[3],self.trans[5] = self.trans[5],self.trans[3]

    def output(self,pos,no):
        pos = np.array(pos).dot(self.trans[no])
        # print (pos)
        return self.model(torch.from_numpy(pos).float()).detach().numpy()
    def posture(self,poss):
        res = []
        for i,p in enumerate(poss):
            res.append(self.output(p,i))
        return res




def querry(inpt,no):
    return fc.output(inpt,no)

if __name__ == "__main__":

    use_cuda = torch.cuda.is_available()
    # use_cuda = False
    device = torch.device("cuda" if use_cuda else "cpu")
    fc = FCnet().to(device)
    # optimizer = optim.SGD(fc.parameters(), lr=0.0001, momentum=0.2,weight_decay=0.00005)
    optimizer = optim.SGD(fc.parameters(), lr=0.0001, momentum=0.02,weight_decay=0.0000005)
    batch_size = 50
    # train
    train_epoch = 400
    logger = Logger("./logs")
    dts = []
    for i in range(3,10):
        with open("data%d.pkl" %i,"rb") as f:
            dts.append( pkl.load(f))
    # with open("data4.pkl","rb") as f:
    #     dt3 = pkl.load(f)
    loss_F = torch.nn.MSELoss()
    # setleglength(clientID,pole,dt[0][1])
    train_dataset = MyDataset(dts)
    # print(train_dataset[0])
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
            x, y = input_data
            # y, x = input_data

            x = x.to(device).float()
            y = y.to(device).float()
            
            pred_y = fc(x)

            loss = loss_F(pred_y, y) # 计算loss
            
            if step %50 == 49: # 每50步，计算精度
                
                print("{}/{} steps".format(step, len(train_dataloader)), float(loss))
                info = { 'loss': float(loss)}
                for tag, value in info.items():
                    logger.scalar_summary(tag, value, epoch*len(train_dataloader)+step)
            # print(x[:2], pred_y[:2])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    print("SAVED! ", 'firstshot4.pt')
    torch.save(fc.state_dict(), 'firstshot4.pt')


fc = footControler("firstshot4.pt")