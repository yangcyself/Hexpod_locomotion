import torch
import torch.autograd as autograd
import torch.nn as nn


class Policy(nn.Module):
    def __init__(self, num_inputs, num_outputs):
        super(Policy, self).__init__()
        self.affine1 = nn.Linear(num_inputs, 2048)
        self.affine2 = nn.Linear(2048, 256)
        self.affine3 = nn.Linear(256, 32)

        self.action_mean = nn.Linear(32, num_outputs)
        self.action_mean.weight.data.mul_(0.1)
        self.action_mean.bias.data.mul_(0.0)

        self.action_log_std = nn.Parameter(torch.zeros(1, num_outputs))

        self.saved_actions = []
        self.rewards = []
        self.final_value = 0

    def forward(self, x):
        x = torch.tanh(self.affine1(x))
        x = torch.tanh(self.affine2(x))
        x = torch.tanh(self.affine3(x))
        action_mean = self.action_mean(x)
        action_log_std = self.action_log_std.expand_as(action_mean)
        action_std = torch.exp(action_log_std)

        return action_mean, action_log_std, action_std


class Value(nn.Module):
    def __init__(self, num_inputs):
        super(Value, self).__init__()
        self.affine1 = nn.Linear(num_inputs, 2048)
        self.affine2 = nn.Linear(2048, 256)
        self.affine3 = nn.Linear(256, 256)
        self.value_head = nn.Linear(256, 1)
        self.value_head.weight.data.mul_(0.1)
        self.value_head.bias.data.mul_(0.0)

    def forward(self, x):
        x = torch.tanh(self.affine1(x))
        x = torch.tanh(self.affine2(x))
        x = torch.tanh(self.affine3(x))
        state_values = self.value_head(x)
        return state_values
