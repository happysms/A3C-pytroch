import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
from torch import tensor
from torch.autograd import Variable

import os
import sys
import time
import cv2
from collections import deque
import numpy as np
import ray

import pandas as pd

# 하이퍼파라미터
learning_rate = 0.0002
gamma = 0.98
n_rollout = 10
print_interval = 20

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class A3C(nn.Module):
    def __init__(self, num_actions):
        super(A3C, self).__init__()

        self.num_actions = num_actions

        self.conv1 = nn.Conv2d(3, 32, kernel_size=8, stride=4)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.bn3 = nn.BatchNorm2d(64)

        def conv2d_size_out(size, kernel_size=3, stride=2):
            return (size - (kernel_size - 1) - 1) // stride + 1

        convw = conv2d_size_out(210, 8, 4)
        convw = conv2d_size_out(convw, 4, 2)
        convw = conv2d_size_out(convw, 3, 1)

        convh = conv2d_size_out(160, 8, 4)
        convh = conv2d_size_out(convh, 4, 2)
        convh = conv2d_size_out(convh, 3, 1)

        linear_input_size = convw * convh * 64
        print("linear_input_size", linear_input_size)
        self.lstm_i_dim = 512  # input dimension of LSTM
        self.lstm_h_dim = 512  # output dimension of LSTM
        self.lstm_N_layer = 1  # number of layers of LSTM
        self.Conv2LSTM = nn.Linear(linear_input_size, self.lstm_i_dim)
        self.lstm = nn.LSTM(input_size=self.lstm_i_dim, hidden_size=self.lstm_h_dim, num_layers=self.lstm_N_layer)

        self.fc_pi = nn.Linear(self.lstm_h_dim, self.num_actions)
        self.fc_v = nn.Linear(self.lstm_h_dim, 1)

    def pi(self, x, softmax_dim=1):
        x = self.fc_pi(x)
        prob = F.softmax(x, dim=softmax_dim)
        return prob

    def v(self, x):
        v = self.fc_v(x)
        return v

    def forward(self, x, hidden, softmax_dim=2):
        x = x/255.0
        if (len(x.shape) < 4):  # 배치학습이 아닐 때
            x = x.unsqueeze(0)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = x.contiguous()  # x torch.Size([1, 64, 22, 16])
        x = x.view(x.size(0), -1)  # x torch Size([1, 22528])
        x = F.relu(self.Conv2LSTM(x))
        x = x.unsqueeze(1)  # x torch Size([1,1,1024])
        x, new_hidden = self.lstm(x, hidden)
        return x, new_hidden


class Agent():
    def __init__(self, model, env, lr, gamma, batch_size, max_epi, agent_num, save_path):
        # Network, Optimizer
        self.model = model
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        # Hyperparmeters
        self.lr = lr
        self.batch_size = batch_size
        self.gamma = gamma

        # env
        self.env = env
        self.print_interval = 20
        self.score = 0.0
        self.max_epi = max_epi
        self.agent_num = agent_num

        # data for n-step training
        self.data = []

        self.save_path = save_path

    def put_data(self, item):
        self.data.append(item)

    def make_batch(self):
        s_lst, a_lst, r_lst, s_prime_lst, done_lst = [], [], [], [], []
        for transition in self.data:
            s, a, r, s_prime, done = transition
            s_lst.append(s)
            a_lst.append([a])
            r_lst.append([r])
            s_prime_lst.append(s_prime)
            done_mask = 0.0 if done else 1.0
            done_lst.append([done_mask])
        s_batch = torch.stack(s_lst).float().to(device)
        a_batch = torch.tensor(a_lst).to(device)
        r_batch = torch.tensor(r_lst).float().to(device)
        s_prime_batch = torch.stack(s_prime_lst).float().to(device)
        done_batch = torch.tensor(done_lst).float().to(device)

        self.data = []
        return s_batch, a_batch, r_batch, s_prime_batch, done_batch

    def update(self, hidden):
        s, a, r, s_prime, done = self.make_batch()  # all tensors size must be [10,1]
        x_prime, _ = self.model.forward(s_prime, hidden)
        v_prime = self.model.v(x_prime)
        v_prime = v_prime.squeeze(1)
        td_target = r + self.gamma * v_prime * done

        x, _ = self.model.forward(s, hidden)
        v = self.model.v(x)
        v = v.squeeze(1)

        delta = td_target - v
        pi = self.model.pi(x, softmax_dim=2)
        a = a.unsqueeze(1)
        pi_a = pi.gather(2, a)

        # Policy Loss + Value Loss
        loss = -torch.log(pi_a) * delta.detach() + F.smooth_l1_loss(v, td_target.detach())

        torch.autograd.set_detect_anomaly(True)
        self.optimizer.zero_grad()
        loss.mean().backward
        self.optimizer.step()

        return loss.mean().item()

    def train(self):
        Episodes = [x for x in range(self.max_epi)]
        train_stats = pd.DataFrame(index=Episodes, columns=['Train loss', 'Rewards'])
        for n_epi in range(self.max_epi):
            done = False
            s = self.env.reset()
            s = torch.from_numpy(s).permute(2, 0, 1).to(device)
            hidden = (Variable(torch.zeros(1, 1, 512).float().to(device=device)),
                      Variable(torch.zeros(1, 1, 512).float().to(device=device)))
            self.score = 0.0
            while not done:
                self.env.render()
                for t in range(n_rollout):
                    x, hidden = self.model.forward(s.float(), hidden)
                    prob = self.model.pi(x, softmax_dim=2)
                    m = Categorical(prob)
                    a = m.sample().item()

                    s_prime, r, done, info = self.env.step(a)
                    s_prime = torch.from_numpy(s_prime).permute(2, 0, 1).to(device)
                    self.put_data((s, a, r, s_prime, done)) # 데이터를 쌓는 부분

                    s = s_prime
                    self.score += r

                    if done:
                        print("%d episode is done" % n_epi)
                        torch.save({'model_state_dict': self.model.state_dict(),
                                    }, self.save_path + 'a3c_lstm.pth')
                        print("model saved")
                        break
                loss = self.update(hidden)
                train_stats.loc[n_epi]['Train loss'] = loss
                train_stats.loc[n_epi]['Rewards'] = self.score
                train_stats.to_csv('train_stat Breakout-v4.csv' + str(self.agent_num))
            print("# of episode :{}, score : {:.1f}".format(n_epi, self.score))
            self.score = 0.0

        self.env.close()

# 로스, 보상 시각화하는 부분 저장할 것
def main():
    env = gym.make("Breakout-v4")


    lr = 0.0005
    gamma = 0.98
    batch_size = 32
    buffer_limit = 50000
    max_epi = 100000
    agent_num = 1
    save_path = os.curdir
    model = A3C(num_actions=4).to(device)
    if(os.path.isfile(save_path+'a3c_lstm.pth')):
        print("Load pretrained model")
        checkpoint = torch.load(save_path + 'a3c_lstm.pth')
        model.load_state_dict(checkpoint['model_state_dict'])

    agent = Agent(model, env, lr, gamma, batch_size, max_epi, agent_num, save_path)
    agent.train()



main()