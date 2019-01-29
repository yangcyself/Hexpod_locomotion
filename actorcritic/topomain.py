# -^- coding:utf-8 -^-
from __future__ import division
# import gym
import numpy as np
import torch
from torch.autograd import Variable
import os
# import psutil
import gc
from logger import Logger
import train
import buffer
# import sys
# sys.path.append("../")
import toyenv as env
from config import*


MAX_EPISODES = 5000
MAX_STEPS = 100
MAX_BUFFER = 1000000
MAX_TOTAL_REWARD = 300


# S_DIM = env.observation_space.shape[0]
# A_DIM = env.action_space.shape[0]
# A_MAX = env.action_space.high[0]

S_DIM = 1616
A_DIM = 6
# A_MAX = 0.3
A_MAX = 0.25


print (' State Dimensions :- ', S_DIM)
print (' Action Dimensions :- ', A_DIM)
print (' Action Max :- ', A_MAX)

ram = buffer.MemoryBuffer(MAX_BUFFER)
trainer = train.Trainer(S_DIM, A_DIM, A_MAX, ram)
logger = Logger("./logs")
logRate = 100
if(RESUME):
    trainer.load_models(RESUME)


def main():
    total_reward = 0
    averagetotoal_reward = 0
    for _ep in range(1,MAX_EPISODES):
        (obs,tpo) = env.reset()
        observation = obs+list(tpo.reshape(-1,))
        _ep = _ep+RESUME
        print("last total reward:", total_reward)
        averagetotoal_reward += total_reward
        total_reward = 0
        print ('EPISODE :- ', _ep)
        for r in range(MAX_STEPS):

            state = np.float32(observation)

            action = trainer.get_exploration_action(state)

            # new_observation, reward, done, info = env.step(action)
            (obs,tpo), reward, done, info = env.step(action)
            total_reward += reward
            new_observation = obs+list(tpo.reshape(-1,))

            if done:
                new_state = None
            else:
                new_state = np.float32(new_observation) 
                # print("HERE",state, action, reward, new_state)
                ram.add(state, action, reward, new_state)

            observation = new_observation

            # perform optimization
            trainer.optimize()
            if done:
                break

        # check memory consumption and clear memory
        gc.collect()

        if _ep%logRate == 0:
            trainer.save_models(_ep)

            info = { 'averageTotalReward': averagetotoal_reward/logRate}

            for tag, value in info.items():
                logger.scalar_summary(tag, value, _ep)

            averagetotoal_reward = 0

    print ('Completed episodes')

if __name__ == "__main__":
    main()
    # env.recover( n=30)
    # env.walk_a_step(0.3,1.57)
    # env.turn_a_deg(0.5)
    