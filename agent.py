import numpy as np
import random
from collections import deque
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.layers import Dense, ReLU, Flatten
from tensorflow.python.keras import Sequential
from tensorflow.python.keras.optimizers import Adam

ACTIONS = {0: (-1, 0), 1: (1, 0), 2: (0, -1), 3: (0, 1)}

class Agent(object):
    def __init__(self, state_size, action_size, alpha=0.7):
        self.state_history = []
        self.alpha = alpha
        self.batch_size = 4
        self.replayBuffer = ExperienceReplay(10, self.batch_size)
        self.n_actions = action_size
        self.lr = 0.001

        self.model = Sequential([
            Dense(units=24,input_dim=state_size, activation = 'relu'),
            Dense(units=24,activation = 'relu'),
            Dense(units=action_size, activation = 'linear')
        ])
        self.model.compile(loss="mse",
                      optimizer = adam(lr=self.lr))
        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_decay = 0.0005

    

    def greedy_policy(self, state):
        q_values = self.model.predict(state)[0]
        return np.argmax(q_values)
        
    
    def epsilon_greedy_policy(self, state, epsilon):
        random_num = random.uniform(0,1)
        if random_num > epsilon:
            action = self.greedy_policy(state)
        else:
            action = np.random.choice(range(self.n_actions))

        return action
    
    #for readability
    def choose_action(self, Qtable, state, epsilon, action_space):
        return self.epsilon_greedy_policy(Qtable, state, epsilon, action_space)
    
    def update_exploration_probability(self):
        self.epsilon = self.epsilon * np.exp(-self.epsilon_decay)

    def add_experience(self, state, action, reward, new_state, done):
        self.replayBuffer.add_experience(state, action, reward, new_state, done)
    
    def train(self):
        # We shuffle the memory buffer and select a batch size of experiences
        batch_sample = self.replayBuffer.sample()
        
        # We iterate over the selected experiences
        for experience in batch_sample:
            # We compute the Q-values of S_t
            q_current_state = self.model.predict(experience["current_state"])
            # We compute the Q-target using Bellman optimality equation
            q_target = experience["reward"]
            if not experience["done"]:
                q_target = q_target + self.gamma*np.max(self.model.predict(experience["next_state"])[0])
            q_current_state[0][experience["action"]] = q_target
            # train the model
            self.model.fit(experience["current_state"], q_current_state, verbose=0)



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


    
    
    