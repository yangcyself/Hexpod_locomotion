
import gym
from dueling_dqn import DuelingDQN
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

import sys
sys.path.append("../")
import env



MEMORY_SIZE = 3000
ACTION_SPACE = 4
actions = np.array([[0,0.1],[0,-0.1],[0.1,0],[-0.1,0]])

sess = tf.Session()
state_dim = 1746
# with tf.variable_scope('natural'):
#     natural_DQN = DuelingDQN(
#         n_actions=ACTION_SPACE, n_features=state_dim, memory_size=MEMORY_SIZE,
#         e_greedy_increment=0.001, sess=sess, dueling=False)

with tf.variable_scope('dueling'):
    dueling_DQN = DuelingDQN(
        n_actions=ACTION_SPACE, n_features=state_dim, memory_size=MEMORY_SIZE,
        e_greedy_increment=0.001, sess=sess, dueling=True, output_graph=True)

sess.run(tf.global_variables_initializer())

class Logger(object):
    
    def __init__(self, log_dir):
        """Create a summary writer logging to log_dir."""
        self.writer = tf.summary.FileWriter(log_dir)

    def scalar_summary(self, tag, value, step):
        """Log a scalar variable."""
        summary = tf.Summary(value=[tf.Summary.Value(tag=tag, simple_value=value)])
        self.writer.add_summary(summary, step)

saver=tf.train.Saver(max_to_keep=1)
logger = Logger("./logs")

def train(RL):
    acc_r = [0]
    total_steps = 0
    episode = 0
    all_reward = 0
    # observation = env.reset()
    while True:
        # if total_steps-MEMORY_SIZE > 9000: env.render()
        s,t = env.reset()
        observation = s+list(t.reshape(-1,))
        for i in range(200):
            action = RL.choose_action(observation)

        # f_action = (action-(ACTION_SPACE-1)/2)/((ACTION_SPACE-1)/4)   # [-2 ~ 2] float actions
            (s_,t), reward, done, info = env.step(actions[action])
            observation_ = s_+list(t.reshape(-1,))
            acc_r.append(reward + acc_r[-1])  # accumulated reward

            RL.store_transition(observation, action, reward, observation_)

            observation = observation_
            total_steps += 1
            all_reward += reward

            if total_steps > MEMORY_SIZE:
                RL.learn()

            if done: 
                break


        # if total_steps-MEMORY_SIZE > 15000:
        #     break
        episode += 1

        if(episode %100 ==  0):
            info = {'averageTotalReward': all_reward / 100}
            all_reward = 0
            for tag, value in info.items():
                logger.scalar_summary(tag, value, i)
            saver.save(sess, './ddpg.ckpt', global_step=episode + 1)
        if(episode>2000):
            break
    return RL.cost_his, acc_r

# c_natural, r_natural = train(natural_DQN)
c_dueling, r_dueling = train(dueling_DQN)

plt.figure(1)
plt.plot(np.array(c_natural), c='r', label='natural')
plt.plot(np.array(c_dueling), c='b', label='dueling')
plt.legend(loc='best')
plt.ylabel('cost')
plt.xlabel('training steps')
plt.grid()

plt.figure(2)
plt.plot(np.array(r_natural), c='r', label='natural')
plt.plot(np.array(r_dueling), c='b', label='dueling')
plt.legend(loc='best')
plt.ylabel('accumulated reward')
plt.xlabel('training steps')
plt.grid()

plt.show()