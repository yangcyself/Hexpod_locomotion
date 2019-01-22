from __future__ import division
# import gym
import numpy as np
import torch
from torch.autograd import Variable
import os
# import psutil
import gc

import train
import buffer
# import sys
# sys.path.append("../")
import toyenv as env


MAX_EPISODES = 5000
MAX_STEPS = 100
MAX_BUFFER = 1000000
MAX_TOTAL_REWARD = 300

# S_DIM = env.observation_space.shape[0]
# A_DIM = env.action_space.shape[0]
# A_MAX = env.action_space.high[0]

S_DIM = 16
A_DIM = 6
# A_MAX = 0.3
A_MAX = 0.25


print (' State Dimensions :- ', S_DIM)
print (' Action Dimensions :- ', A_DIM)
print (' Action Max :- ', A_MAX)

ram = buffer.MemoryBuffer(MAX_BUFFER)
trainer = train.Trainer(S_DIM, A_DIM, A_MAX, ram)
RESUME = 1000
# RESUME = 0
if(RESUME):
    trainer.load_models(RESUME)

def main():
    for _ep in range(MAX_EPISODES):
        observation = env.reset()
        _ep = _ep+RESUME

        print ('EPISODE :- ', _ep)
        for r in range(MAX_STEPS):

            state = np.float32(observation)

            action = trainer.get_exploration_action(state)

            new_observation, reward, done, info = env.step(action)


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

        if _ep%100 == 0:
            trainer.save_models(_ep)

    print ('Completed episodes')

if __name__ == "__main__":
    main()
    # env.recover( n=30)
    # env.walk_a_step(0.3,1.57)
    # env.turn_a_deg(0.5)
    