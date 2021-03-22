import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
from torch import tensor
from torch.autograd import Variable

import os
import time
import cv2
import numpy as np
import ray
from copy import deepcopy
import pandas as pd

global T
global T_max
T = 0
T_max = 10000000


# model construction
class A3C(nn.Module):
    def __init__(self, num_actions):
        super(A3C, self).__init__()

        self.num_actions = num_actions

        self.conv1 = nn.Conv2d(3, 8, kernel_size=8, stride=4)
        self.bn1 = nn.BatchNorm2d(8)
        self.conv2 = nn.Conv2d(8, 16, kernel_size=4, stride=2)
        self.bn2 = nn.BatchNorm2d(16)
        self.conv3 = nn.Conv2d(16, 32, kernel_size=3, stride=1)
        self.bn3 = nn.BatchNorm2d(32)

        def conv2d_size_out(size, kernel_size=3, stride=2):
            return (size - (kernel_size - 1) - 1) // stride + 1

        convw = conv2d_size_out(210, 8, 4)
        convw = conv2d_size_out(convw, 4, 2)
        convw = conv2d_size_out(convw, 3, 1)

        convh = conv2d_size_out(160, 8, 4)
        convh = conv2d_size_out(convh, 4, 2)
        convh = conv2d_size_out(convh, 3, 1)

        linear_input_size = convw * convh * 32
        print("linear_input_size", linear_input_size)
        self.lstm_i_dim = 16  # input dimension of LSTM
        self.lstm_h_dim = 16  # output dimension of LSTM
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


@ray.remote
class Actor_Learner():

    def __init__(self, model, env, lr, gamma, max_epi, agent_num, save_path):
        # Network, Optimizer
        print("address", id(model))
        self.learner_model = model.cuda()
        self.actor_model = model.cpu()
        self.optimizer = optim.Adam(self.learner_model.parameters(), lr=lr)

        # Hyperparmeters
        self.lr = lr
        self.gamma = gamma

        # thread counter
        self.t = 1
        self.t_max = 4

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
        device = 'cpu'
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

    def calculate_loss(self, hidden):
        s, a, r, s_prime, done = self.make_batch()  # all tensors size must be [10,1]
        x_prime, _ = self.actor_model.forward(s_prime, hidden)
        v_prime = self.actor_model.v(x_prime)
        v_prime = v_prime.squeeze(1)
        td_target = r + self.gamma * v_prime * done

        x, _ = self.actor_model.forward(s, hidden)
        v = self.actor_model.v(x)
        v = v.squeeze(1)

        delta = td_target - v
        pi = self.actor_model.pi(x, softmax_dim=2)
        a = a.unsqueeze(1)
        pi_a = pi.gather(2, a)

        # Policy Loss + Value Loss
        loss = -torch.log(pi_a) * delta.detach() + F.smooth_l1_loss(v, td_target.detach())
        loss.mean().backward(retain_graph=True)

        return loss.mean().item()

    def accumulate_gradients(self):
        for actor_net, learner_net in zip(self.actor_model.named_parameters(), self.learner_model.named_parameters()):
            learner_net[1].grad = deepcopy(actor_net[1].grad)
            #print(learner_net[1].grad)

    def train(self):
        global T
        global T_max

        Episodes = [x for x in range(self.max_epi)]
        train_stats = pd.DataFrame(index=Episodes, columns=['Train loss', 'Rewards'])
        if(T < T_max):
            for n_epi in range(self.max_epi):
                # Reset gradients and Synchronize thread params with global params
                self.optimizer.zero_grad()
                self.actor_model = deepcopy(self.learner_model).cpu()
                loss = 0
                # state initialiaztion
                device = 'cpu'
                done = False
                s = self.env.reset()
                s = torch.from_numpy(s).permute(2, 0, 1).to(device)
                hidden = (Variable(torch.zeros(1, 1, 16).float().to(device=device)),
                          Variable(torch.zeros(1, 1, 16).float().to(device=device)))

                while not done:
                    self.env.render()
                    for t in range(self.t_max):
                        x, hidden = self.actor_model.forward(s.float(), hidden)
                        prob = self.actor_model.pi(x, softmax_dim=2)
                        m = Categorical(prob)
                        a = m.sample().item()
                        s_prime, r, done, info = self.env.step(a)
                        s_prime = torch.from_numpy(s_prime).permute(2, 0, 1).to(device)
                        self.put_data((s, a, r, s_prime, done))  # 데이터를 쌓는 부분

                        s = s_prime
                        self.score += r
                        T += 1

                        if done:
                            break
                    loss = self.calculate_loss(hidden)
                    self.accumulate_gradients()
                    self.optimizer.step()

                train_stats.loc[n_epi]['Train loss'] = loss
                train_stats.loc[n_epi]['Rewards'] = self.score
                train_stats.to_csv('train_stat Breakout-v4.csv' + str(self.agent_num))
                print("# of episode :{}, score : {:.1f}".format(n_epi, self.score))
                self.save_model()
                self.score = 0.0
            self.env.close()


    def save_model(self):
        torch.save({'model_state_dict': self.learner_model.state_dict()}, self.save_path + 'a3c_lstm2.pth')
        print("model saved")


# Multi agent

def main():
    env = gym.make("Breakout-v4")
    ray.shutdown()
    ray.init()
    lr = 0.0005
    gamma = 0.98
    max_epi = 100000
    save_path = os.curdir

    model = A3C(num_actions=4)
    shared_model = ray.put(model)
    agent1 = Actor_Learner.remote(shared_model, env, lr, gamma, max_epi, 1, save_path)
    agent2 = Actor_Learner.remote(shared_model, env, lr, gamma, max_epi, 2, save_path)
    #agent3 = Actor_Learner.remote(shared_model, env, lr, gamma, max_epi, 3, save_path)
    #agent4 = Actor_Learner.remote(shared_model, env, lr, gamma, max_epi, 4, save_path)
    #agent5 = Actor_Learner.remote(shared_model, env, lr, gamma, max_epi, 5, save_path)
    #agent6 = Actor_Learner.remote(shared_model, env, lr, gamma, max_epi, 6, save_path)
    #agent7 = Actor_Learner.remote(shared_model, env, lr, gamma, max_epi, 7, save_path)
    #agent8 = Actor_Learner.remote(shared_model, env, lr, gamma, max_epi, 8, save_path)

    result = [agent1.train.remote(),
              agent2.train.remote(),
              #agent3.train.remote(),
              #agent4.train.remote()
              #agent5.train.remote(),
              #agent6.train.remote(),
              #agent7.train.remote(),
              #agent8.train.remote()
              ]

    ray.get(result)

    env.close()


main()
