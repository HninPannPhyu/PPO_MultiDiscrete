import os
import numpy as np
import torch as T
import torch.nn as nn
import torch.optim as optim
from torch.distributions.categorical import Categorical


class PPOMemory:
    def __init__(self, batch_size):
        self.states = []
        self.probs_0 = []
        self.probs_1 = []
        self.vals = []
        self.actions_0 = []
        self.actions_1 = []
        self.rewards = []
        self.dones = []

        self.batch_size = batch_size

    def generate_batches(self):
        n_states = len(self.states)
        batch_start = np.arange(0, n_states, self.batch_size)
        indices = np.arange(n_states, dtype=np.int64)
        np.random.shuffle(indices)
        batches = [indices[i:i + self.batch_size] for i in batch_start]

        return np.array(self.states), \
            np.array(self.actions_0), \
            np.array(self.actions_1), \
            np.array(self.probs_0), \
            np.array(self.probs_1), \
            np.array(self.vals), \
            np.array(self.rewards), \
            np.array(self.dones), \
            batches

    def store_memory(self, state, action, probs, vals, reward, done):
        self.states.append(state)
        self.actions_0.append(action[0])
        self.actions_1.append(action[1])
        self.probs_0.append(probs[0])
        self.probs_1.append(probs[1])
        self.vals.append(vals)
        self.rewards.append(reward)
        self.dones.append(done)

    def clear_memory(self):
        self.states = []
        self.probs_0 = []
        self.probs_1 = []
        self.actions_0 = []
        self.actions_1 = []
        self.rewards = []
        self.dones = []
        self.vals = []


class ActorNetwork(nn.Module): #create different action for each action from Multi Discrete
    def __init__(self, n_actions, input_dims, alpha,
                 fc1_dims=256, fc2_dims=256, chkpt_dir=r'C:\Users\AR15960\PycharmProjects\RL_customEnv_and_algos'):
        super(ActorNetwork, self).__init__()

        self.checkpoint_file = os.path.join(chkpt_dir, 'actor_torch_ppo')
        self.actor_0 = nn.Sequential( #create different action for each action from Multi Discrete
            nn.Linear(*input_dims, fc1_dims),
            nn.ReLU(),
            nn.Linear(fc1_dims, fc2_dims),
            nn.ReLU(),
            nn.Linear(fc2_dims, n_actions[0]),
            nn.Softmax(dim=-1)
        )
        self.actor_1 = nn.Sequential(#create different action for each action from Multi Discrete
            nn.Linear(*input_dims, fc1_dims),
            nn.ReLU(),
            nn.Linear(fc1_dims, fc2_dims),
            nn.ReLU(),
            nn.Linear(fc2_dims, n_actions[1]),
            nn.Softmax(dim=-1)
        )

        self.optimizer = optim.Adam(self.parameters(), lr=alpha)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state):
        dist0 = self.actor_0(state)
        dist0 = Categorical(dist0)
        # print(f'dist: {dist}')
        dist1 = self.actor_1(state)
        dist1 = Categorical(dist1)

        return dist0,dist1

    def save_checkpoint(self):
        T.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        self.load_state_dict(T.load(self.checkpoint_file))


class CriticNetwork(nn.Module):
    def __init__(self, input_dims, alpha, fc1_dims=256, fc2_dims=256,
                 chkpt_dir=r'C:\Users\AR15960\PycharmProjects\RL_customEnv_and_algos'):
        super(CriticNetwork, self).__init__()

        self.checkpoint_file = os.path.join(chkpt_dir, 'critic_torch_ppo')
        self.critic = nn.Sequential(
            nn.Linear(*input_dims, fc1_dims),
            nn.ReLU(),
            nn.Linear(fc1_dims, fc2_dims),
            nn.ReLU(),
            nn.Linear(fc2_dims, 1)
        )

        self.optimizer = optim.Adam(self.parameters(), lr=alpha)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state):
        value = self.critic(state)

        return value

    def save_checkpoint(self):
        T.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        self.load_state_dict(T.load(self.checkpoint_file))


class Agent:
    def __init__(self, n_actions, input_dims, gamma=0.99, alpha=0.0003, gae_lambda=0.95,
                 policy_clip=0.2, batch_size=64, n_epochs=10):
        self.gamma = gamma
        self.policy_clip = policy_clip
        self.n_epochs = n_epochs
        self.gae_lambda = gae_lambda

        self.actor = ActorNetwork(n_actions, input_dims, alpha)
        self.critic = CriticNetwork(input_dims, alpha)
        self.memory = PPOMemory(batch_size)

    def remember(self, state, action, probs, vals, reward, done):
        self.memory.store_memory(state, action, probs, vals, reward, done)

    def save_models(self):
        print('... saving models ...')
        self.actor.save_checkpoint()
        self.critic.save_checkpoint()

    def load_models(self):
        print('... loading models ...')
        self.actor.load_checkpoint()
        self.critic.load_checkpoint()

    def choose_action(self, observation):
        print(f'observation{observation}')
        state = T.tensor([observation], dtype=T.float).to(self.actor.device)

        dist0,dist1 = self.actor(state)
        value = self.critic(state)
        action_0 = dist0.sample()
        action_1= dist1.sample()

        probs0 = T.squeeze(dist0.log_prob(action_0)).item()
        action_0 = T.squeeze(action_0).item()
        probs1 = T.squeeze(dist1.log_prob(action_1)).item()
        action_1 = T.squeeze(action_1).item()
        action=np.array([action_0,action_1])
        probs=np.array([probs0,probs1])
        # print(f'action:{action}')
        value = T.squeeze(value).item()

        return action, probs, value

    def learn(self):
        for _ in range(self.n_epochs):
            state_arr, action_arr0, action_arr1,old_prob_arr0,old_prob_arr1, vals_arr, \
                reward_arr, dones_arr, batches = \
                self.memory.generate_batches()

            values = vals_arr
            advantage = np.zeros(len(reward_arr), dtype=np.float32)

            for t in range(len(reward_arr) - 1):
                discount = 1
                a_t = 0
                for k in range(t, len(reward_arr) - 1):
                    a_t += discount * (reward_arr[k] + self.gamma * values[k + 1] * \
                                       (1 - int(dones_arr[k])) - values[k])
                    discount *= self.gamma * self.gae_lambda
                advantage[t] = a_t
            advantage = T.tensor(advantage).to(self.actor.device)

            values = T.tensor(values).to(self.actor.device)
            for batch in batches:
                states = T.tensor(state_arr[batch], dtype=T.float).to(self.actor.device)
                old_probs0 = T.tensor(old_prob_arr0[batch]).to(self.actor.device)
                old_probs1 = T.tensor(old_prob_arr1[batch]).to(self.actor.device)
                actions0 = T.tensor(action_arr0[batch]).to(self.actor.device)
                actions1= T.tensor(action_arr1[batch]).to(self.actor.device)

                dist0,dist1 = self.actor(states)
                critic_value = self.critic(states)
                # print(f'dist0  {dist0}')
                # print(f'dist0  {dist1}')
                # print(f'actions {actions0}')

                critic_value = T.squeeze(critic_value)

                new_probs0 = dist0.log_prob(actions0)

                prob_ratio_0 = new_probs0.exp() / old_probs0.exp()

                new_probs1 = dist1.log_prob(actions1)
                prob_ratio_1 = new_probs1.exp() / old_probs1.exp()

                prob_ratio=(prob_ratio_0+prob_ratio_1)/2
                # prob_ratio = (new_probs - old_probs).exp()
                weighted_probs = advantage[batch] * prob_ratio
                weighted_clipped_probs = T.clamp(prob_ratio, 1 - self.policy_clip,
                                                 1 + self.policy_clip) * advantage[batch]
                actor_loss = -T.min(weighted_probs, weighted_clipped_probs).mean()

                returns = advantage[batch] + values[batch]
                critic_loss = (returns - critic_value) ** 2
                critic_loss = critic_loss.mean()

                total_loss = actor_loss + 0.5 * critic_loss
                self.actor.optimizer.zero_grad()
                self.critic.optimizer.zero_grad()
                total_loss.backward()
                self.actor.optimizer.step()
                self.critic.optimizer.step()

        self.memory.clear_memory()


