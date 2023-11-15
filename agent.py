import numpy as np
import random
ACTIONS = {0: (-1, 0), 1: (1, 0), 2: (0, -1), 3: (0, 1)}

class Agent(object):
    def __init__(self, states, alpha=0.7):
        self.state_history = []
        self.alpha = alpha
        
        # start the rewards table
        self.G = {}
        self.init_reward(states)

    def init_reward(self, states):
        for i, row in enumerate(states):
            for j, col in enumerate(row):
                self.G[(i, j)] = 0
                # np.random.uniform(low=1.0, high=0.1)

    def greedy_policy(self, state, action_space):
        max_val  = -10e15
        best_action = None
        for action in action_space:
            new_state = tuple([sum(x) for x in zip(state, ACTIONS[action])])
            if self.G[new_state] >= max_val:
                best_action = action
                max_val = self.G[new_state]
        
        return best_action
        
        
    
    def epsilon_greedy_policy(self, state, epsilon, action_space):
        # Randomly generate a number between 0 and 1
        random_num = random.uniform(0,1)
        # if random_num > greater than epsilon --> exploitation
        if random_num > epsilon:
            # Take the action with the highest value given a state
            action = self.greedy_policy(state, action_space)
            # print("learned action")
        # else --> exploration
        else:
            action = np.random.choice(action_space)
            # print("random action")

        return action
    
        #for readability
    def choose_action(self, state, epsilon, action_space):
        return self.epsilon_greedy_policy(state, epsilon, action_space)
    
    def update_state_history(self, state, reward):
        self.state_history.append((state, reward))

    
    def learn(self):
        target = 0

        for prev, reward in reversed(self.state_history):
            self.G[prev] = self.G[prev] + self.alpha * (target - self.G[prev])
            target += reward

        self.state_history = []

    
    
    
