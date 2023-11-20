import numpy as np
from agent import Agent
from environment import Maze
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import seaborn as sns



def visualize_q_values(agent, num_rows, num_columns):
    # Assuming your Q-values are stored in the agent's model
    # and can be accessed for each state
    q_values = np.zeros((num_rows, num_columns, len(ACTIONS)))

    for row in range(num_rows):
        for col in range(num_columns):
            state = row * num_columns + col
            state_tensor = torch.tensor([state], dtype=torch.int64)
            state_tensor = F.one_hot(state_tensor, num_classes=num_rows*num_columns).float().unsqueeze(0)
            
            with torch.no_grad():
                q = agent.model(state_tensor).numpy()
            q_values[row, col] = q

    for action in ACTIONS:
        plt.figure()
        sns.heatmap(q_values[:, :, action], annot=True, fmt=".2f")
        plt.title(f"Q-values for Action: {action}")
        plt.show()

def visualize_policy(agent, num_rows, num_columns, env):
    policy_grid = np.zeros((num_rows, num_columns), dtype=str)

    for row in range(num_rows):
        for col in range(num_columns):
            state = row * num_columns + col
            state_tensor = torch.tensor([state], dtype=torch.int64)
            state_tensor = F.one_hot(state_tensor, num_classes=num_rows*num_columns).float().unsqueeze(0)

            with torch.no_grad():
                q_values = agent.model(state_tensor)
            best_action = np.argmax(q_values.numpy())

            if env.maze[row, col] == 1 or env.maze[row, col] == 3 or env.maze[row, col] == 2:
                policy_grid[row, col] = env.maze[row, col]
                continue

            # Represent the action in a human-readable way
            if best_action == 0:
                policy_grid[row, col] = 'U'
            elif best_action == 1:
                policy_grid[row, col] = 'D'
            elif best_action == 2:
                policy_grid[row, col] = 'L'
            elif best_action == 3:
                policy_grid[row, col] = 'R'

    print("Policy Grid:")
    print(policy_grid)






ACTIONS = {0: (-1, 0), 1: (1, 0), 2: (0, -1), 3: (0, 1)}
action_to_int = {'U': 0, 'D': 1, 'L': 2, 'R': 3}

environment = """
A00000000x
0000000000
0000000000
0011111000
0010G01000
0010001000
0000000000
0000000000
0000000000
x00000000x
"""

if __name__ == "__main__":
    env = Maze(environment)
    
    max_epsilon = 1.0             # Exploration probability at start
    min_epsilon = 0.05            # Minimum exploration probability
    decay_rate = 0.005            # Exponential decay rate for exploration prob
    agent = Agent(gamma=0.99, epsilon=1.0, lr=0.001, input_dims=[1], n_actions=4, batch_size=64)
    n_training_episodes = 5000
    max_steps = 500
    total_steps = 0
    deaths = 0
    wins = 0
    
  # We iterate over episodes
    for e in range(n_training_episodes):
        # We initialize the first state and reshape it to fit 
        #  with the input layer of the DNN
        if e % 1000 == 0:
            print(e)

        current_state = env.reset()
        for step in range(max_steps):
            total_steps = total_steps + 1
            action = agent.choose_action(current_state)
            while not env.is_allowed_move(current_state, action):
                action = agent.choose_action(current_state)

            next_state, reward, done, _ = env.step(action)
            agent.store_transition(current_state, action, reward, next_state, done)
            
            if done:
                print("Status:", _)
                if _ == "dead":
                    deaths+=1
                elif _ == "goal":
                    wins+=1
                break
            agent.learn()
            current_state = next_state
  
    
    print("Total wins:", wins)
    print("Total deaths:", deaths)
    print("Total timeouts:", n_training_episodes - wins - deaths)
    # Call this function with your agent and environment dimensions
    visualize_policy(agent, env.rows, env.columns, env)


