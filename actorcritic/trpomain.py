import argparse
from itertools import count

# import gym
import scipy.optimize

import torch
from trpo.models import *
from trpo.replay_memory import Memory
from trpo.running_state import ZFilter
from torch.autograd import Variable
from trpo.trpo import trpo_step
from trpo.utils import *
from config import *
from logger import Logger,Tlogger
import finalenv as env

torch.utils.backcompat.broadcast_warning.enabled = True
torch.utils.backcompat.keepdim_warning.enabled = True

torch.set_default_tensor_type('torch.DoubleTensor')

parser = argparse.ArgumentParser(description='PyTorch actor-critic example')
parser.add_argument('--gamma', type=float, default=0.995, metavar='G',
                    help='discount factor (default: 0.995)')
parser.add_argument('--env-name', default="Reacher-v1", metavar='G',
                    help='name of the environment to run')
parser.add_argument('--tau', type=float, default=0.97, metavar='G',
                    help='gae (default: 0.97)')
parser.add_argument('--l2-reg', type=float, default=1e-3, metavar='G',
                    help='l2 regularization regression (default: 1e-3)')
parser.add_argument('--max-kl', type=float, default=1e-2, metavar='G',
                    help='max kl value (default: 1e-2)')
parser.add_argument('--damping', type=float, default=1e-1, metavar='G',
                    help='damping (default: 1e-1)')
parser.add_argument('--seed', type=int, default=543, metavar='N',
                    help='random seed (default: 1)')
parser.add_argument('--batch-size', type=int, default=1500, metavar='N',
                    help='random seed (default: 1)')
parser.add_argument('--render', action='store_true',
                    help='render the environment')
parser.add_argument('--log-interval', type=int, default=4, metavar='N',
                    help='interval between training status logs (default: 10)')
args = parser.parse_args()

# env = gym.make(args.env_name)

# num_inputs = env.observation_space.shape[0]
# num_actions = env.action_space.shape[0]

num_inputs = env.observation_space.shape[0]
if(OBSERVETOPO):
    num_inputs += 1600
if(FUTHERTOPO):
    num_inputs += 144

num_actions = env.action_space.shape[0]

# env.seed(args.seed)
# torch.manual_seed(args.seed)


def select_action(state):
    state = torch.from_numpy(state).unsqueeze(0)
    action_mean, _, action_std = policy_net(Variable(state))
    action = torch.normal(action_mean, action_std)
    action = (action/torch.sum(action*action))/2
    # action = torch.clamp(action,min = -1,max  = )
    return action

def update_params(batch):
    rewards = torch.Tensor(batch.reward)
    masks = torch.Tensor(batch.mask)
    actions = torch.Tensor(np.concatenate(batch.action, 0))
    states = torch.Tensor(batch.state)
    values = value_net(Variable(states))

    returns = torch.Tensor(actions.size(0),1)
    deltas = torch.Tensor(actions.size(0),1)
    advantages = torch.Tensor(actions.size(0),1)

    prev_return = 0
    prev_value = 0
    prev_advantage = 0
    for i in reversed(range(rewards.size(0))):
        returns[i] = rewards[i] + args.gamma * prev_return * masks[i]
        deltas[i] = rewards[i] + args.gamma * prev_value * masks[i] - values.data[i]
        advantages[i] = deltas[i] + args.gamma * args.tau * prev_advantage * masks[i]

        prev_return = returns[i, 0]
        prev_value = values.data[i, 0]
        prev_advantage = advantages[i, 0]

    targets = Variable(returns)

    # Original code uses the same LBFGS to optimize the value loss
    def get_value_loss(flat_params):
        set_flat_params_to(value_net, torch.Tensor(flat_params))
        for param in value_net.parameters():
            if param.grad is not None:
                param.grad.data.fill_(0)

        values_ = value_net(Variable(states))

        value_loss = (values_ - targets).pow(2).mean()

        # weight decay
        for param in value_net.parameters():
            value_loss += param.pow(2).sum() * args.l2_reg
        value_loss.backward()
        return (value_loss.data.double().numpy(), get_flat_grad_from(value_net).data.double().numpy())

    flat_params, _, opt_info = scipy.optimize.fmin_l_bfgs_b(get_value_loss, get_flat_params_from(value_net).double().numpy(), maxiter=25)
    set_flat_params_to(value_net, torch.Tensor(flat_params))

    advantages = (advantages - advantages.mean()) / advantages.std()

    action_means, action_log_stds, action_stds = policy_net(Variable(states))
    fixed_log_prob = normal_log_density(Variable(actions), action_means, action_log_stds, action_stds).data.clone()

    def get_loss(volatile=False):
        if volatile:
            with torch.no_grad():
                action_means, action_log_stds, action_stds = policy_net(Variable(states))
        else:
            action_means, action_log_stds, action_stds = policy_net(Variable(states))
                
        log_prob = normal_log_density(Variable(actions), action_means, action_log_stds, action_stds)
        action_loss = -Variable(advantages) * torch.exp(log_prob - Variable(fixed_log_prob))
        return action_loss.mean()


    def get_kl():
        mean1, log_std1, std1 = policy_net(Variable(states))

        mean0 = Variable(mean1.data)
        log_std0 = Variable(log_std1.data)
        std0 = Variable(std1.data)
        kl = log_std1 - log_std0 + (std0.pow(2) + (mean0 - mean1).pow(2)) / (2.0 * std1.pow(2)) - 0.5
        return kl.sum(1, keepdim=True)

    trpo_step(policy_net, get_loss, get_kl, args.max_kl, args.damping)

# running_state = ZFilter((num_inputs,), clip=5)

# running_reward = ZFilter((1,), demean=False, clip=10)

policy_net = Policy(num_inputs, num_actions)
value_net = Value(num_inputs)
if(RESUME):
        
    policy_net.load_state_dict(torch.load('./Models/' + str(RESUME) + '_actor.pt'))
    value_net.load_state_dict(torch.load('./Models/' + str(RESUME) + '_critic.pt'))


logger = Logger("./logs")

for i_episode in count(1):
    memory = Memory()

    num_steps = 0
    reward_batch = 0
    num_episodes = 0

    i_episode += RESUME
    print("EPISODE:",i_episode)
    while num_steps < args.batch_size:
        if(OBSERVETOPO):
            state,tpo = env.reset()
            state = np.array(state+list(tpo.reshape(-1,)))
        else:
            state = np.array(env.reset())
        # state = running_state(state)
        
        reward_sum = 0
        print("num_steps:",num_steps)
        for t in range(100): # Don't infinite loop while learning

            action = select_action(state)
            action = action.data[0].numpy()
            print("reward_sum:",reward_sum)
            if(OBSERVETOPO):
                (obs,tpo), reward, done, _ = env.step(action)
                next_state = np.array(obs+list(tpo.reshape(-1,)))
            else:
                next_state,reward, done, _ = env.step(action)
                next_state = np.array(next_state)
            reward_sum += reward
            # next_state = running_state(next_state)
            # print(next_state)
            mask = 1
            if done:
                mask = 0

            memory.push(state, np.array([action]), mask, next_state, reward)

            # if args.render:
            #     env.render()
            if done:
                break

            state = next_state
        
        num_steps += (t-1)
        num_episodes += 1
        reward_batch += reward_sum
        Tlogger.refresh()

    reward_batch /= num_episodes
    batch = memory.sample()
    update_params(batch)
    if i_episode % args.log_interval == 0:
        print('Episode {}\tLast reward: {}\tAverage reward {:.2f}'.format(
            i_episode, reward_sum, reward_batch))

        info = { 'Average reward': reward_batch}
        for tag, value in info.items():
            logger.scalar_summary(tag, value, i_episode)
        Tlogger.refresh()
        torch.save(policy_net.state_dict(), './Models/' + str(i_episode) + '_actor.pt')
        torch.save(value_net.state_dict(), './Models/' + str(i_episode) + '_critic.pt')
        print ('Models saved successfully')

