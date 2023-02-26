
import gym
import math
import random
import numpy as np
from collections import namedtuple, deque
from itertools import count
from torch.distributions import MultivariateNormal
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

env = gym.make('Pendulum-v1')
env.reset(seed = 543)
torch.manual_seed(seed = 543)
n_obs = env.observation_space.shape[0]
n_act = env.action_space.shape[0]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
reward_threshold =  0
print(reward_threshold)

class reinforce(nn.Module):    
    def __init__(self, n_observations, n_actions):
        super(reinforce, self).__init__()
        self.layer1  = nn.Linear(n_observations, 128)
        self.layer2 = nn.Linear(128, 128)
        self.layer3 = nn.Linear(128, n_actions)  
        
        self.saved_log_probs = []
        self.saved_log_probs_curr = []
        self.rewards = []
        self.obs = []
        self.act = []
    def forward(self, x):
        act1 = F.relu(self.layer1(x))
        act2 = F.relu(self.layer2(act1))
        act3 = F.relu(act2)
        action_scores = self.layer3(act3)
        return F.softmax(action_scores, dim = 1) 
    
policy = reinforce(n_obs,n_act)
crits = reinforce(n_obs, 1)

optimizer = optim.Adam(policy.parameters(), lr=1e-2)
crits_optimizer = optim.Adam(crits.parameters(), lr=1e-2)
eps = np.finfo(np.float32).eps.item()

cov_var = torch.full(size=(n_act,), fill_value=0.5)
cov_mat = torch.diag(cov_var)

def policy_selection(state):
    state = torch.from_numpy(state).float().unsqueeze(0)
    probs = policy(state)
    m = MultivariateNormal(probs,cov_mat)
    action = m.sample()
    policy.saved_log_probs.append(m.log_prob(action))
    return action.detach().numpy()[0,:]

def eval(batch_obs, batch_acts):
    V = crits(batch_obs)
    probs2 = policy(batch_obs)
    m2 = MultivariateNormal(probs2,cov_mat)
    loggy = m2.log_prob(batch_acts)
    #crits.saved_log_probs_curr.append(m2.log_prob(batch_acts))
    return V,loggy

def terminate():
    R = 0
    clip = 0.2
    gamma = 0.95
    policy_loss = []
    returns = deque()
    
    obs = deque()
    acts = deque()
    log1 = deque()
    log2 = deque()
    for r in policy.rewards[::-1]:
        R = r + gamma * R
        returns.appendleft(R)

    
    for k in crits.act:
        acts.appendleft(k)
        
    for l in crits.obs:
        obs.appendleft(l)

    returns = torch.tensor(returns)
    
    obs = np.array(obs)
    obs = torch.tensor(obs)

    acts = np.array(acts)
    acts = torch.tensor(acts)
    Kritik,log_rew = eval(obs, acts)
    Kritik = Kritik.to(torch.float64)
    ret = returns.unsqueeze(1)
    A_k = ret - Kritik
    
    A_k = (A_k - A_k.mean()) / (A_k.std() + eps)
        
    prev_log = torch.tensor(policy.saved_log_probs)

    ratios = torch.exp(log_rew - prev_log)

    surr1 = ratios * A_k
    surr2 = torch.clamp(ratios, 1 - clip, 1 + clip) * A_k

    actor_loss = (-torch.min(surr1, surr2)).mean()
    
    loss = nn.MSELoss()
    critic_loss = loss(Kritik.squeeze(), returns)

    optimizer.zero_grad()
    actor_loss.backward(retain_graph=True)
    optimizer.step()

    crits_optimizer.zero_grad()
    critic_loss.backward()
    crits_optimizer.step()
    
    
running_reward = 0

for i in count():
    state = env.reset()
    ep_reward = 0
    kun = deque()
    tul = deque()
    #obs = []
    for t in range(1, 1000):
        #env.render()
        action = policy_selection(state)
        state, reward, done, info = env.step(action)
        
        policy.rewards.append(reward)
        crits.rewards.append(reward)
        crits.act.append(action)
        ep_reward += np.array(policy.rewards[-1])
        crits.obs.append(state)

        if done:
            break
    #print(len(policy.rewards))
    reward_episodic = ep_reward / (len(policy.rewards) + 1e-6 )
    running_reward = 0.05 *( ep_reward / (len(policy.rewards) + 1e-6 ))+ (1 - 0.05) * running_reward
    
    terminate()
    if i % 10 == 0:
        print("Episode number:", i/10, "Prev. Reward:", reward_episodic, "Ave. Reward:", running_reward)
        torch.save(policy.state_dict(), './ppo_actor.pth')
        torch.save(crits.state_dict(), './ppo_critic.pth')
    if reward_episodic > reward_threshold:
    #     #env.close()
        print("Done!")
        break