from __future__ import division
# import gym
import sys
import numpy as np
import torch
from torch.autograd import Variable
import os
# import psutil
import gc
from logger import Logger,Tlogger
import train
import buffer
from os import listdir
import string
# import sys
# sys.path.append("../")
import toyenv as env
# import oldenv as env
from config import*
import pickle as pkl

with open("cmpRes.pkl","rb") as f:
    cmpRes = pkl.load(f)

MAX_EPISODES = 50000
MAX_STEPS = 35
MAX_BUFFER = 10000
MAX_TOTAL_REWARD = 300

S_DIM = 1615 #1600 + 160 + 15
if(FUTHERTOPO):
    S_DIM += 144
A_DIM = 6
A_MAX = 0.25

CMPLIST = [100,500,600,1000,1500,2000,3000,3500,4000,5000,6000,7000,8000,9000,10000,11000,12000,13000,14000,15000]#,17000,19000,22000,25000,28000,30000]
CMPMODS = ["hopior/","savior/"]
print (' State Dimensions :- ', S_DIM)
print (' Action Dimensions :- ', A_DIM)
print (' Action Max :- ', A_MAX)

ram = buffer.MemoryBuffer(MAX_BUFFER)
trainer = train.Trainer(S_DIM, A_DIM, A_MAX, ram)

for M in CMPMODS:
    if(M not in cmpRes.keys()):
        cmpRes[M] = {}
    for N in CMPLIST:
        print(M,N,end = "\t")
        if(N in cmpRes[M].keys()):
            continue
        try:
            trainer.load_models(M+str(N))
        except:
            continue
        (obs,tpo) = env.reset()
        observation = obs+list(tpo.reshape(-1,))
        total_reward = 0
        for r in range(MAX_STEPS):

            state = np.float32(observation)

            action = trainer.get_exploitation_action(state)    
            (obs,tpo), reward, done, info = env.step(action)
            total_reward += reward
            new_observation = obs+list(tpo.reshape(-1,))

            if done:
                new_state = None
            else:
                new_state = np.float32(new_observation) 
                # print("HERE",state, action, reward, new_state)

            observation = new_observation
            if done:
                break
        print(total_reward)
        cmpRes[M][N] = total_reward

with open("cmpRes.pkl","wb") as f:
    pkl.dump(cmpRes,f)