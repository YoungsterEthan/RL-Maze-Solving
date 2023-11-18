import numpy as np
import random
from collections import deque
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
# Define a simple neural network in PyTorch equivalent to the TensorFlow code provided
class DQNN(nn.Module):
    def __init__(self, state_size, action_size):
        super(DQNN, self).__init__()
        self.layer1 = nn.Linear(state_size, 24)  # First dense layer with 24 units
        self.layer2 = nn.Linear(24, 24)          # Second dense layer with 24 units
        self.output_layer = nn.Linear(24, action_size)  # Output layer

    def forward(self, x):
        x = torch.relu(self.layer1(x))  # Apply ReLU activation to the first layer
        x = torch.relu(self.layer2(x))  # Apply ReLU activation to the second layer
        x = self.output_layer(x)        # Linear activation for the output layer
        return x

ACTIONS = {0: (-1, 0), 1: (1, 0), 2: (0, -1), 3: (0, 1)}

class Agent(object):
    def __init__(self, state_size, action_size, alpha=0.7):
        self.state_history = []
        self.alpha = alpha
        self.batch_size = 64
        self.replayBuffer = ExperienceReplay(1000, self.batch_size)
        self.n_actions = action_size
        self.lr = 0.001
        self.model = DQNN(state_size, action_size)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        self.loss_function = torch.nn.MSELoss()

        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_decay = 0.0005

    

    # def greedy_policy(self, state):
    #     state_tensor = torch.tensor(state, dtype=torch.float32)
    #     q_values = F.one_hot(state_tensor.long(), num_classes=100).float()
    #     q_current = self.model(q_values)

    #     return torch.max(q_current)
        
    
    def epsilon_greedy_policy(self, state):
        # Directly convert the state to a tensor
        state_tensor = torch.tensor(state, dtype=torch.int64)  # Convert state to tensor
        state_tensor = F.one_hot(state_tensor, num_classes=100).float().unsqueeze(0)  # One-hot encoding

        random_num = random.uniform(0,1)
        if random_num > self.epsilon:
            with torch.no_grad():  # Turn off gradient tracking for inference
                q_values = self.model(state_tensor)
            action = torch.argmax(q_values).item()  # Get the action with the highest Q-value
        else:
            action = np.random.choice(range(self.n_actions))

        return action

    
    #for readability
    def choose_action(self, state):
        return self.epsilon_greedy_policy(state)
    
    def update_exploration_probability(self):
        self.epsilon = self.epsilon * np.exp(-self.epsilon_decay)

    def add_experience(self, state, action, reward, new_state, done):
        self.replayBuffer.add_experience(state, action, reward, new_state, done)
    
    def train(self):
        self.model.train()
        for experience in self.replayBuffer.sample():
            state = torch.tensor(experience["state"], dtype=torch.int64)
            state = F.one_hot(state, num_classes=100).float().unsqueeze(0)

            next_state = torch.tensor(experience["new_state"], dtype=torch.int64)
            next_state = F.one_hot(next_state, num_classes=100).float().unsqueeze(0)

            action = experience["action"]
            reward = experience["reward"]
            done = experience["done"]

            q_current = self.model(state)
            q_next = self.model(next_state).detach()
            q_target = q_current.clone()

            q_target[0][0][action] = reward + self.gamma * torch.max(q_next) * (not done)

            self.optimizer.zero_grad()
            loss = self.loss_function(q_current, q_target)
            loss.backward()
            self.optimizer.step()





class ExperienceReplay:
    def __init__(self, max_size, batch_size):
        self.buffer = deque(maxlen=max_size)
        self.max_size = max_size
        self.batch_size = batch_size

    def add_experience(self, state, action, reward, new_state, done):
        # Add experience to the buffer. Older experiences are automatically removed if max_size is reached.
        self.buffer.append({"state":state, "action":action, "reward":reward, "new_state":new_state, "done":done})

    def sample(self):
        # Sample a batch of experiences from the buffer
        batch_size = min(self.batch_size, len(self.buffer))  # Ensures not to sample more than buffer size
        sampled_experiences = random.sample(self.buffer, batch_size)
        return sampled_experiences


    
    
    