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

        
class MLP(nn.Module):
    def __init__(self, state_size, action_size, INTEGRATION_RIGHT_LIMIT):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(state_size, 64)
        if INTEGRATION_RIGHT_LIMIT == -1.0:
            self.fc2 = nn.Linear(64, 64)
        else:
            self.fc2 = MLP_ODE(INTEGRATION_RIGHT_LIMIT)
        self.fc3 = nn.Linear(64, action_size)

    def forward(self, state):
        res = F.relu(self.fc1(state))
        res = F.relu(self.fc2(res))
        return self.fc3(res)

        
class MLP_ODE(nn.Module):
    def __init__(self, INTEGRATION_RIGHT_LIMIT):
        super(MLP_ODE, self).__init__()
        self.network = ConcatLinear(64, 64)
        self.integration_time = torch.tensor([0, INTEGRATION_RIGHT_LIMIT]).float()
    
    def forward(self, state):
        self.integration_time = self.integration_time.type_as(state)
        out = odeint(self.network, state, self.integration_time, method='euler')
        return out[1] 

        
class dqn():
    def __init__(self, state_size, action_size, device, buffer_size=100000, batch_size=256, gamma=0.99, lr=5e-4, replace_target=100, INTEGRATION_RIGHT_LIMIT=1):
        self.state_size = state_size
        self.action_size = action_size
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.gamma = gamma
        self.lr = lr
        self.replace_target = replace_target
        self.INTEGRATION_RIGHT_LIMIT = INTEGRATION_RIGHT_LIMIT
        
        self.device = device
        
        self.local_model = MLP(state_size, action_size, INTEGRATION_RIGHT_LIMIT).to(self.device)
        self.target_model = MLP(state_size, action_size, INTEGRATION_RIGHT_LIMIT).to(self.device)
        self.optimizer = optim.Adam(self.local_model.parameters(), lr=lr)
        
        self.memory = memory_buffer(self.buffer_size, self.batch_size)
        self.time = 0
    
    def update(self, transition):
        state, action, reward, next_state, done = transition
        self.memory.add(state, action, reward, next_state, done)
        self.time += 1
        if len(self.memory) > self.batch_size:
            batch = self.memory.sample()
            
            states = torch.from_numpy(np.vstack([e[0] for e in batch if e is not None])).float().to(self.device)
            actions = torch.from_numpy(np.vstack([e[1] for e in batch if e is not None])).long().to(self.device)
            rewards = torch.from_numpy(np.vstack([e[2] for e in batch if e is not None])).float().to(self.device)
            next_states = torch.from_numpy(np.vstack([e[3] for e in batch if e is not None])).float().to(self.device)
            dones = torch.from_numpy(np.vstack([e[4] for e in batch if e is not None])).to(self.device)

            #with torch.no_grad():
            #    target_pred = rewards + ((~dones) * self.gamma * self.target_model(next_states).detach().max(1)[0].unsqueeze(1))
            
            
            # DDQN
            with torch.no_grad():
                amax = self.local_model(next_states).detach().max(1)[1].unsqueeze(1)
                target_pred = rewards + ((~dones) * self.gamma * self.target_model(next_states).detach().gather(1, amax))
            # DDQN
                
            local_pred = self.local_model(states).gather(1, actions)

            loss = F.mse_loss(local_pred, target_pred)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            if self.time % self.replace_target == 0:
                self.target_model.load_state_dict(self.local_model.state_dict())  

    def act(self, state):
        state = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
        with torch.no_grad():
            action_values = self.local_model(state)
        return np.argmax(action_values.cpu().data.numpy())

    def save(self, path='agent.pkl'):
        torch.save(self.local_model.state_dict(), path)
        
    def check_model(self, episodes=100):
        history = []
        local_env = gym.make('LunarLander-v2')
        for _ in range(episodes):
            state = local_env.reset()
            done = False
            total = 0
            
            while not done:
                action = self.act(state)
                next_state, reward, done, _ = local_env.step(action)
                state = next_state
                total += reward  
                
            history.append(total)
            
        history = np.array(history)
        
        return history
     
     
class memory_buffer:
    def __init__(self, max_len, batch_size):
        self.mem = [None] * max_len
        self.max_len = max_len
        self.tail = 0
        self.len = 0
        self.batch_size = batch_size
    
    def add(self, state, action, reward, next_state, done):
        self.mem[self.tail] = (state, action, reward, next_state, done)
        self.len = min(self.len + 1, self.max_len)
        self.tail = (self.tail + 1) % self.max_len
    
    def sample(self):
        temp = random.sample(range(self.len), self.batch_size)
        return [self.mem[i] for i in temp]

    def __len__(self):
        return self.len
        

def train_loop(args):
    id, time_const, device_name = args
    INTEGRATION_RIGHT_LIMIT = time_const
    
    environment = 'LunarLander-v2'
    env = gym.make('LunarLander-v2')
    
    seed = 228
    env.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    
    episodes = 3000
    eps_border = 1e-7
    decay_rate = 0.997
    eps = 1.0
    
    state_size = 8
    action_size = 4
    
    check_episodes = 100
    threshold = 220
    
    device = torch.device(device_name)
    
    agent = dqn(state_size=state_size, action_size=action_size, device=device, INTEGRATION_RIGHT_LIMIT=INTEGRATION_RIGHT_LIMIT)
    history = deque(maxlen=100)
    
    for episode in range(episodes):
        print(episode)
        state = env.reset()
        score = 0
        eps = max(eps_border, eps * decay_rate)
        done = False
        while not done:
            if random.random() < eps:
                action = random.choice(np.arange(action_size))
            else:
                action = agent.act(state)
            next_state, reward, done, _ = env.step(action)
            transition = state, action, reward, next_state, done
            agent.update(transition)
            state = next_state
            score += reward  
        history.append(score)
        if episode % 25 == 0:
            local_history = agent.check_model(episodes=check_episodes)
            local_mean = np.mean(local_history)
            local_var = np.sqrt(np.var(local_history))
            writer.add_scalar(f'id_{id}_mean', local_mean, episode)
            writer.add_scalar(f'id_{id}_var', local_var, episode)
            writer.flush()
            agent.save(path=f'id_{id}_agent_{episode}.pkl')

        
if __name__ == "__main__":
    writer = SummaryWriter("output")
    
    print('Begin!')
    
    int_time_list = [(1, -1.0, 'cuda:0'),
    (2, 0.1, 'cuda:1'),
    (3, 0.5, 'cuda:0'),
    (4, 1.0, 'cuda:1'), 
    (5, 3.0, 'cuda:0'), 
    (6, 5.0, 'cuda:1'),
    (7, 10.0, 'cuda:0')]
    
    p = multiprocessing.Pool(processes=22)
    
    #train_loop((1, -1.0, 'cuda:0'))
    
    
    p.map(train_loop, int_time_list)
    
    p.close()
    p.join()
    
    writer.close()
    
    print('Done!')
