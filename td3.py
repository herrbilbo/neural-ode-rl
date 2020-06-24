import gym
import random
import numpy as np
from collections import deque
import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn
import pybullet_envs

from torchdiffeq import odeint

from torch.utils.tensorboard import SummaryWriter

import multiprocessing.dummy as multiprocessing


class ConcatLinear(nn.Module):
    def __init__(self, dim_in, dim_out):
        super(ConcatLinear, self).__init__()
        self._layer = nn.Linear(dim_in + 1, dim_out)
        nn.init.xavier_normal_(self._layer.weight)
        self._layer.bias.data.fill_(0.01)

    def forward(self, t, x):
        tt = torch.ones_like(x[:, :1]) * t
        ttx = torch.cat([tt, x], 1)
        return self._layer(ttx)
        
        
class MLP_ODE(nn.Module):
    def __init__(self, layer_size, INTEGRATION_RIGHT_LIMIT):
        super(MLP_ODE, self).__init__()
        self.network = ConcatLinear(layer_size, layer_size)
        self.integration_time = torch.tensor([0, INTEGRATION_RIGHT_LIMIT]).float()
    
    def forward(self, state):
        self.integration_time = self.integration_time.type_as(state)
        out = odeint(self.network, state, self.integration_time, method='euler')
        return out[1] 
        
        
class actor(nn.Module):
    def __init__(self, state_size, action_size, layer_size, INTEGRATION_RIGHT_LIMIT):
        super(actor, self).__init__()
        self.fc1 = nn.Linear(state_size, layer_size)
        
        if INTEGRATION_RIGHT_LIMIT == -1.0:
            self.fc2 = nn.Linear(layer_size, layer_size)
            nn.init.xavier_normal_(self.fc2.weight)
            self.fc2.bias.data.fill_(0.01)
        else:
            self.fc2 = MLP_ODE(layer_size, INTEGRATION_RIGHT_LIMIT)
            
        self.fc3 = nn.Linear(layer_size, action_size)
        nn.init.xavier_normal_(self.fc1.weight)
        nn.init.xavier_normal_(self.fc3.weight)
        self.fc1.bias.data.fill_(0.01)
        self.fc3.bias.data.fill_(0.01)

    def forward(self, state):
        res = F.relu(self.fc1(state))
        res = F.relu(self.fc2(res))
        res = torch.tanh(self.fc3(res))
        return res
        
        
class critic(nn.Module):
    def __init__(self, state_size, action_size, layer_size, INTEGRATION_RIGHT_LIMIT):
        super(critic, self).__init__()
        self.fc1 = nn.Linear(state_size + action_size, layer_size)
        if INTEGRATION_RIGHT_LIMIT == -1.0:
            self.fc2 = nn.Linear(layer_size, layer_size)
            nn.init.xavier_normal_(self.fc2.weight)
            self.fc2.bias.data.fill_(0.01)
        else:
            self.fc2 = MLP_ODE(layer_size, INTEGRATION_RIGHT_LIMIT)
        self.fc3 = nn.Linear(layer_size, 1)
        
        self.fc4 = nn.Linear(state_size + action_size, layer_size)
        if INTEGRATION_RIGHT_LIMIT == -1.0:
            self.fc5 = nn.Linear(layer_size, layer_size)
            nn.init.xavier_normal_(self.fc5.weight)
            self.fc5.bias.data.fill_(0.01)
        else:
            self.fc5 = MLP_ODE(layer_size, INTEGRATION_RIGHT_LIMIT)
        self.fc6 = nn.Linear(layer_size, 1)
        
        nn.init.xavier_normal_(self.fc1.weight)
        nn.init.xavier_normal_(self.fc3.weight)
        nn.init.xavier_normal_(self.fc4.weight)
        nn.init.xavier_normal_(self.fc6.weight)
        self.fc1.bias.data.fill_(0.01)
        self.fc3.bias.data.fill_(0.01)
        self.fc4.bias.data.fill_(0.01)
        self.fc6.bias.data.fill_(0.01)
        
    def critic_1(self, state, action):
        res = torch.cat((state, action), dim=1)
        res = F.relu(self.fc1(res))
        res = F.relu(self.fc2(res))
        res = self.fc3(res)
        return res
        
    def critic_2(self, state, action):
        res = torch.cat((state, action), dim=1)
        res = F.relu(self.fc4(res))
        res = F.relu(self.fc5(res))
        res = self.fc6(res)
        return res

    def forward(self, state, action):
        return (self.critic_1(state, action), self.critic_2(state, action))
        
        
class replay_buffer:
    def __init__(self, max_size, batch_size):
        self.max_size = max_size
        self.batch_size = batch_size
        self.buffer = deque(maxlen=max_size)

    def push(self, transition):
        self.buffer.append(transition)

    def sample(self):
        return list(zip(*random.sample(self.buffer, self.batch_size)))

    def __len__(self):
        return len(self.buffer)
        
        
class td3():
    def __init__(self, environment_name, state_dim, action_dim, buffer_size, batch_size, gamma, tau, actor_lr, critic_lr, std, std_min, std_decay, c, update_every, sigma, layer_size, INTEGRATION_RIGHT_LIMIT, device):
        self.environment_name = environment_name
        
        self.device = device
        
        self.gamma = gamma
        self.tau = tau
        
        self.actor_lr = actor_lr
        self.critic_lr = critic_lr
        
        self.std = std
        self.std_min = std_min
        self.std_decay = std_decay
        
        self.c = c
        self.update_every = update_every
        self.sigma = sigma
        self.cur_time = 0
        
        self.actor = actor(state_dim, action_dim, layer_size=layer_size, INTEGRATION_RIGHT_LIMIT=INTEGRATION_RIGHT_LIMIT).to(self.device)
        self.critic = critic(state_dim, action_dim, layer_size=layer_size, INTEGRATION_RIGHT_LIMIT=INTEGRATION_RIGHT_LIMIT).to(self.device)
        
        self.actor_target = actor(state_dim, action_dim, layer_size=layer_size, INTEGRATION_RIGHT_LIMIT=INTEGRATION_RIGHT_LIMIT).to(self.device)
        self.critic_target = critic(state_dim, action_dim, layer_size=layer_size, INTEGRATION_RIGHT_LIMIT=INTEGRATION_RIGHT_LIMIT).to(self.device)
        
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=self.actor_lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=self.critic_lr)

        self.hard_update()

        self.replay_buffer = replay_buffer(buffer_size, batch_size)
    
    def update(self, transition):
            self.replay_buffer.push(transition)
            
            if len(self.replay_buffer) >= self.replay_buffer.batch_size:
                self.cur_time += 1
                batch = self.replay_buffer.sample()
                
                states, actions, rewards, next_states, dones = batch

                states = torch.tensor(states).to(self.device).float()
                next_states = torch.tensor(next_states).to(self.device).float()
                rewards = torch.tensor(rewards).to(self.device).float()
                actions = torch.tensor(actions).to(self.device).float()
                dones = torch.tensor(dones).to(self.device).int()

                with torch.no_grad():
                    next_actions = self.actor_target(next_states)
                    noise = ((torch.randn_like(actions) * self.sigma).clamp(-self.c, self.c)).to(self.device)
                    next_actions = (next_actions + noise).clamp(-1, 1).float()
                    
                    Q_target1, Q_target2 = self.critic_target(next_states, next_actions)            
                    Q_target = rewards.unsqueeze(1) + (self.gamma * torch.min(Q_target1, Q_target2) * ((1 - dones).unsqueeze(1)))
                    
                critic_1, critic_2 = self.critic(states, actions)
                #critic_loss = (critic_1 - Q_target) ** 2 + (critic_2 - Q_target) ** 2
                critic_loss = F.mse_loss(critic_1, Q_target) + F.mse_loss(critic_2, Q_target)                
                
                self.critic_optimizer.zero_grad()
                critic_loss.backward()
                self.critic_optimizer.step()

                if self.cur_time % self.update_every == 0:
                    actor_loss = -self.critic.critic_1(states, self.actor(states)).mean()

                    self.actor_optimizer.zero_grad()
                    actor_loss.backward()
                    self.actor_optimizer.step()
                    self.soft_update()

    def act(self, state, noise=False):
        state = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
        with torch.no_grad():
            action = self.actor(state).cpu().data.numpy()
            
        if noise:
            noise = np.random.normal(loc=0.0, scale=self.std, size=action.shape)
            action = action + noise
            action = np.clip(action, -1.0, 1.0)
            self.std = max(self.std - self.std_decay, self.std_min)
            
        return action[0]
        
    def hard_update(self):
        for target_param, param in zip(self.actor_target.parameters(), self.actor.parameters()):
            target_param.data.copy_(param.data)
            
        for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
            target_param.data.copy_(param.data)

    def soft_update(self):
        for target_param, param in zip(self.actor_target.parameters(), self.actor.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
            
        for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

    def save(self, path='gg'):
        torch.save(self.actor.state_dict(), path + '_actor.pkl')
        torch.save(self.critic.state_dict(), path + '_critic.pkl')
        
    def check_model(self, episodes=100):
        history = []
        local_env = gym.make(self.environment_name)
        for _ in range(episodes):
            state = local_env.reset()
            done = False
            total = 0
            
            while not done:
                action = self.act(state, noise=False)
                next_state, reward, done, _ = local_env.step(action)
                state = next_state
                total += reward  
                
            history.append(total)
            
        history = np.array(history)
        
        return history
        

def train_loop(args):
    id, time_const, seed, device_name = args
    INTEGRATION_RIGHT_LIMIT = time_const
    
    environment_name = 'Walker2DBulletEnv-v0'
    
    env = gym.make(environment_name)
    
    #seed = 228
    env.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    
    episodes = 3000
    
    layer_size = 128
    
    state_dim = 22
    action_dim = 6
    
    buffer_size = 50000
    batch_size = 128
    
    gamma = 0.99
    
    actor_lr = 1e-4
    critic_lr = 1e-4
    tau = 0.05
    
    check_episodes = 100
    threshold = 250
    
    std = 0.3
    std_min = 0.05
    std_decay = (std - std_min) / 500.0
    
    c = 0.5
    update_every = 2
    sigma = 0.2
    
    device = torch.device(device_name)
    
    agent = td3(environment_name, 
    state_dim, 
    action_dim, 
    buffer_size, 
    batch_size, 
    gamma, 
    tau, 
    actor_lr, 
    critic_lr, 
    std,
    std_min,
    std_decay,
    c,
    update_every,
    sigma,
    layer_size, 
    INTEGRATION_RIGHT_LIMIT, 
    device)
    
    history = deque(maxlen=25)
    
    for episode in range(episodes):
        state = env.reset()
        score = 0
        done = False
        while not done:
            if episode < 25:
                action = env.action_space.sample()
            else:
                action = agent.act(state, noise=True)
                
            next_state, reward, done, _ = env.step(action)
            transition = state, action, reward, next_state, done
            agent.update(transition)
            state = next_state
            score += reward  
            
        history.append(score)
        if episode % 25 == 0:
            agent.save(path=f'id_{id}_agent_{episode}')
        if episode % 25 == 0:
            local_history = agent.check_model(episodes=check_episodes)
            local_mean = np.mean(local_history)
            local_var = np.sqrt(np.var(local_history))
            writer.add_scalar(f'id_{id}_mean', local_mean, episode)
            writer.add_scalar(f'id_{id}_var', local_var, episode)
            writer.flush()
            #if local_mean >= threshold:
            #    agent.save(path=f'id_{id}_agent_{episode}')

        
if __name__ == "__main__":
    writer = SummaryWriter("output")
    
    print('Begin!')
    # 42, 131, 455, 16
    int_time_list = [(1, 0.1, 42, 'cuda:0'),
    (2, 0.3, 42, 'cuda:1'),
    (3, 0.1, 131, 'cuda:0'),
    (4, 0.3, 131, 'cuda:1'),
    (5, 0.1, 455, 'cuda:0'),
    (6, 0.3, 455, 'cuda:1'),
    (7, 0.1, 16, 'cuda:0'),
    (8, 0.3, 16, 'cuda:1')]
    
    #p = multiprocessing.Pool()
    p = multiprocessing.Pool(processes=22)
    
    
    p.map(train_loop, int_time_list)
    
    p.close()
    p.join()
    
    #train_loop((1, -1.0, 'cuda:0'))
    
    writer.close()
    
    print('Done!')
