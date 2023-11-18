import numpy as np
from agent import Agent
from environment import Maze


ACTIONS = {0: (-1, 0), 1: (1, 0), 2: (0, -1), 3: (0, 1)}
action_to_int = {'U': 0, 'D': 1, 'L': 2, 'R': 3}

# environment = """
# A0001x1000
# 0000101000
# 0000000001
# 1000000000
# 1111000000
# G00x000000
# 000x000100
# 0000000100
# 0000010100
# 000000x000
# """
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

def print_Q(Q):
    for i, state in enumerate(Q):
        y = i // 10
        x = i % 10
        print(f'({y},{x}): {state}')
        


if __name__ == "__main__":
    env = Maze(environment)
    agent = Agent(100,4)
    max_epsilon = 1.0             # Exploration probability at start
    min_epsilon = 0.05            # Minimum exploration probability
    decay_rate = 0.005            # Exponential decay rate for exploration prob
    n_training_episodes = 1000
    max_steps = 500
    total_steps = 0
    deaths = 0
    wins = 0
    
  # We iterate over episodes
    for e in range(n_training_episodes):
        # We initialize the first state and reshape it to fit 
        #  with the input layer of the DNN
        current_state = env.reset()
        current_state = np.array([current_state])
        for step in range(500):
            total_steps = total_steps + 1
            # the agent computes the action to perform
            action = agent.choose_action(current_state)
            while not env.is_allowed_move(current_state, action):
                action = agent.choose_action(current_state)

            # the envrionment runs the action and returns
            # the next state, a reward and whether the agent is done
            next_state, reward, done, _ = env.step(action)
            next_state = np.array([next_state])
            if next_state < 0:
                print("NEXT STATE", next_state)
            
            # We sotre each experience in the memory buffer
            agent.add_experience(current_state, action, reward, next_state, done)
            
            # if the episode is ended, we leave the loop after
            # updating the exploration probability
            if done:
                print("Status:", _)
                if _ == "dead":
                    deaths+=1
                elif _ == "goal":
                    env.print_last_five_states()
                    wins+=1

                # env.print_last_five_states()
                agent.update_exploration_probability()
                break
            current_state = next_state
        # if the have at least batch_size experiences in the memory buffer
        # than we train our model
        if total_steps >= agent.batch_size:
            agent.train()
    
    print("Total wins:", wins)
    print("Total deaths:", deaths)
    print("Total timeouts:", n_training_episodes - wins - deaths)
