import numpy as np
from agent import Agent
from environment import Maze

ACTIONS = {0: (-1, 0), 1: (1, 0), 2: (0, -1), 3: (0, 1)}
action_to_int = {'U': 0, 'D': 1, 'L': 2, 'R': 3}

def train(agent:Agent, env:Maze, Qtable, n_training_episodes, min_epsilon, max_epsilon, decay_rate, max_steps):
    deaths = 0
    wins = 0
    for episode in range(n_training_episodes):
        # Reduce epsilon (because we need less and less exploration)
        epsilon = min_epsilon + (max_epsilon - min_epsilon)*np.exp(-decay_rate*episode)
        # Reset the environment
        state = env.reset()
        step = 0
        gamma = 1
        env.construct_allowed_states()
        action_space = env.allowed_states

        show_result = False
        for step in range(max_steps):
            # Choose the action At using epsilon greedy policy
            allowed_moves = action_space[state]
            action = agent.choose_action(Qtable, state, epsilon, allowed_moves)
            new_state, reward, status, is_over = env.step(action)

            Qtable[state][action] = Qtable[state][action] + agent.alpha * (reward + gamma * np.max(Qtable[new_state]) - Qtable[state][action])

            # print(env.maze)

            if status == "dead":
                deaths += 1
                print("The agent has died at step", step)
                # show_result = True

            elif status == "goal":
                wins +=1
                print("The agent has won at step", step)
                show_result = True
            # If terminated or truncated finish the episode
            if is_over:
                break

            # Our next state is the new state
            state = new_state
        if show_result:
            env.print_last_five_states()

    return agent.G, deaths, wins

environment = """
A0001x1000
0000101000
0000000001
1000000000
1111000000
G00x000000
000x000100
0000000100
0000010100
000000x000
"""
def print_Q(Q):
    for k,v in Q.items():
        print(k, v)


if __name__ == "__main__":
    env = Maze(environment)
    agent = Agent(env.maze)
    Qtable = np.zeros((100, 4))
    max_epsilon = 1.0             # Exploration probability at start
    min_epsilon = 0.05            # Minimum exploration probability
    decay_rate = 0.0005            # Exponential decay rate for exploration prob
    n_training_episodes = 1000
    max_steps = 500
    G, deaths, wins = train(agent, env, Qtable, n_training_episodes, min_epsilon, max_epsilon, decay_rate, max_steps) 
    print_Q(G)
    print('Total deaths:', deaths)
    print('Total wins:', wins)
    print('Total timeouts:', n_training_episodes - deaths - wins)
