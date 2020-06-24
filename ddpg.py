import gym
import random
import numpy as np
from collections import deque
import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn

from torchdiffeq import odeint

from torch.utils.tensorboard import SummaryWriter

import multiprocessing.dummy as multiprocessing


class ConcatLinear(nn.Module):
    def __init__(self, dim_in, dim_out):
        super(ConcatLinear, self).__init__()
        self._layer = nn.Linear(dim_in + 1, dim_out)

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
        else:
            self.fc2 = MLP_ODE(layer_size, INTEGRATION_RIGHT_LIMIT)
            
        self.fc3 = nn.Linear(layer_size, action_size)

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
        else:
            self.fc2 = MLP_ODE(layer_size, INTEGRATION_RIGHT_LIMIT)
        
        self.fc3 = nn.Linear(layer_size, 1)

    def forward(self, state, action):
        res = torch.cat((state, action), dim=1)
        res = F.relu(self.fc1(res))
        res = F.relu(self.fc2(res))
        res = self.fc3(res)
        return res
        
        
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
        
        
class ddpg():
    def __init__(self, environment, state_dim, action_dim, buffer_size, batch_size, gamma, tau, actor_lr, critic_lr, weight_decay, layer_size, INTEGRATION_RIGHT_LIMIT, device):
        self.environment = environment
        
        self.device = device
        
        self.gamma = gamma
        self.tau = tau
        
        self.actor_lr = actor_lr
        self.critic_lr = critic_lr
        
        self.actor = actor(state_dim, action_dim, layer_size=layer_size, INTEGRATION_RIGHT_LIMIT=INTEGRATION_RIGHT_LIMIT).to(self.device)
        self.critic = critic(state_dim, action_dim, layer_size=layer_size, INTEGRATION_RIGHT_LIMIT=INTEGRATION_RIGHT_LIMIT).to(self.device)
        
        self.actor_target = actor(state_dim, action_dim, layer_size=layer_size, INTEGRATION_RIGHT_LIMIT=INTEGRATION_RIGHT_LIMIT).to(self.device)
        self.critic_target = critic(state_dim, action_dim, layer_size=layer_size, INTEGRATION_RIGHT_LIMIT=INTEGRATION_RIGHT_LIMIT).to(self.device)
        
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=self.actor_lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=self.critic_lr, weight_decay=weight_decay)

        self.hard_update()

        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.replay_buffer = replay_buffer(self.buffer_size, self.batch_size)
    
    def update(self, transition):
            self.replay_buffer.push(transition)
            
            if len(self.replay_buffer) >= self.batch_size:
                batch = self.replay_buffer.sample()
                
                states, actions, rewards, next_states, dones = batch

                states = torch.tensor(states).to(self.device).float()
                next_states = torch.tensor(next_states).to(self.device).float()
                rewards = torch.tensor(rewards).to(self.device).float()
                actions = torch.tensor(actions).to(self.device).float()
                dones = torch.tensor(dones).to(self.device).int()

                next_actions = self.actor_target(next_states)
                Q = self.critic_target(next_states, next_actions).detach()
                Q = rewards.unsqueeze(1) + self.gamma * ((1 - dones).unsqueeze(1)) * Q

                critic_loss = ((self.critic(states, actions) - Q)**2).mean()

                self.critic_optimizer.zero_grad()
                critic_loss.backward()
                self.critic_optimizer.step()

                actions_pred = self.actor(states)
                
                actor_loss = -self.critic(states, actions_pred).mean()

                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                self.actor_optimizer.step()

                self.soft_update() 

    def act(self, state):
        state = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
        with torch.no_grad():
            action = self.actor(state).cpu().data.numpy()
        return action
        
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

    def save(self, path=''):
        torch.save(self.actor.state_dict(), path + '_actor.pkl')
        torch.save(self.critic.state_dict(), path + '_critic.pkl')
        
    def check_model(self, episodes=100):
        history = []
        local_env = gym.make(self.environment)
        for _ in range(episodes):
            state = local_env.reset()
            done = False
            total = 0
            
            while not done:
                action = self.act(state)
                action = action[0]
                next_state, reward, done, _ = local_env.step(action)
                state = next_state
                total += reward  
                
            history.append(total)
            
        history = np.array(history)
        
        return history
        

def train_loop(args):
    id, time_const, device_name = args
    INTEGRATION_RIGHT_LIMIT = time_const
    environment = 'BipedalWalker-v3'
    env = gym.make(environment)
    
    seed = 228
    env.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    
    episodes = 5000
    
    layer_size = 256
    
    state_dim = 24
    action_dim = 4
    
    buffer_size = 1000000
    batch_size = 128
    
    gamma = 0.99
    
    actor_lr = 1e-4
    critic_lr = 3e-4
    weight_decay = 1e-4
    tau = 1e-3
    
    check_episodes = 100
    threshold = 250
    
    std_min = 0.01
    std_decay = 0.0005
    
    #device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device_name)
    
    agent = ddpg(environment, 
    state_dim, 
    action_dim, 
    buffer_size, 
    batch_size, 
    gamma, 
    tau, 
    actor_lr, 
    critic_lr, 
    weight_decay, 
    layer_size, 
    INTEGRATION_RIGHT_LIMIT,
    device)
    history = deque(maxlen=100)

    #print()
    #print('*' * 100)
    #print('*' * 100)
    #print('*' * 100)
    #print(f'Starting training loop id={id} with parameters:')
    #print(f'device={agent.device}')
    #print(f'env={environment}')
    #if INTEGRATION_RIGHT_LIMIT != -1:
    #    print(f'integration_time={INTEGRATION_RIGHT_LIMIT}')
    #else:
    #    print(f'integration_time=BASELINE')
    #print(f'layer_size={layer_size}')
    #print(f'episodes={episodes}')
    #print(f'gamma={gamma}')
    #print(f'actor_lr={actor_lr}')
    #print(f'critic_lr={critic_lr}')
    #print(f'weight_decay={weight_decay}')
    #print(f'tau={tau}')
    #print(f'buffer_size={buffer_size}')
    #print(f'batch_size={batch_size}')
    #print(f'noise_std_min={std_min}')
    #print(f'noise_std_decay={std_decay}')
    #print(f'seed={seed}')
    #print(f'check_episodes={check_episodes}')
    #print(f'threshold={threshold}')
    #print('*' * 100)
    #print('*' * 100)
    #print('*' * 100)
    
    std = 1
    
    print(id, agent.device)
    
    for episode in range(1, episodes):
        state = env.reset()
        score = 0
        done = False
        while not done:
            action = agent.act(state)
            
            noise = np.random.normal(loc=0.0, scale=std, size=action.shape)
            action = action + noise
            action = np.clip(action, -1.0, 1.0)
            
            action = action[0]
            
            next_state, reward, done, _ = env.step(action)
            transition = state, action, reward, next_state, done
            agent.update(transition)
            state = next_state
            score += reward  
            
        history.append(score)
        std = max(std - 0.005, std_min)
        #print(f'\repisode {episode}, mean: {np.mean(history)}, variance: {np.sqrt(np.var(history))}, min: {np.min(history)}, max: {np.max(history)}, std: {std}', end="")
        #if episode % 100 == 0:
            #print(f'\repisode {episode}, mean: {np.mean(history)}, variance: {np.sqrt(np.var(history))}, min: {np.min(history)}, max: {np.max(history)}, std: {std}')
        if episode % 100 == 0:
            local_history = agent.check_model(episodes=check_episodes)
            local_mean = np.mean(local_history)
            local_var = np.sqrt(np.var(local_history))
            writer.add_scalar(f'id_{id}_mean', local_mean, episode)
            writer.add_scalar(f'id_{id}_var', local_var, episode)
            writer.flush()
            if local_mean >= threshold:
                #print()
                #print('*' * 100)
                #print(f"I am saving agent on iteration={episode}, it's score is:")
                #print(f"mean={local_mean}")
                #print(f"sqrt(var)={local_var}")
                #print(f"min={np.min(local_history)}")
                #print(f"max={np.max(local_history)}")
                #print('*' * 100)
                agent.save(path=f'id_{id}_agent_{episode}.pkl')

        
if __name__ == "__main__":
    writer = SummaryWriter("output")
    
    print('Begin!')
    
    #int_time_list = [(1, -1.0, 'cuda:0'), (2, 1.0, 'cuda:1'), (3, 0.1, 'cuda:2'), (4, 0.5, 'cuda:3'), (5, 3, 'cuda:0'), (6, 5, 'cuda:1'), (7, 10, 'cuda:2')]
    int_time_list = [(1, -1.0, 'cuda:0'), (2, 1.0, 'cuda:1')]
    
    p = multiprocessing.Pool(processes=22)
    
    p.map(train_loop, int_time_list)
    
    p.close()
    p.join()
    
    writer.close()
    
    print('Done!')