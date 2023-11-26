import numpy as np
from agent import DDQNAgent
from environment import Maze
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import seaborn as sns
from utils import plotLearning
import time
ACTIONS = {0: (0, -1), 1: (1, 0), 2: (0, 1), 3: (-1, 0)}
action_to_int = {'L': 0, 'D': 1, 'R': 2, 'U': 3}

#TODO ADD PENALTY FOR ATTEMPTING TO MOVE IN AN BLOCKED DIRECTION (HITTING A WALL) OR INVALID DIRECTION (ATTEMPTING TO GO OUT OF BOUNDS)

environment_one = """
A000000000
0000000000
0000000000
0011111000
0010G01000
0010001000
0000000000
0000000000
0000000000
0000000000
"""

environment_two = """
A00000000x
0000000000
0000000000
0011111000
0010G01000
0010x01000
00x000x000
0000000000
0000000000
x00000000x
"""

environment_three = """
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


def train(agent, env:Maze, n_episodes, n_steps):
    deaths = 0
    wins = 0
    scores = []
    eps_history = []
    for i in range(n_episodes):
        history = []
        score = 0
        step = 0
        done = False
        observation = env.reset()
        while not done and step < n_steps:
            action = agent.choose_action(observation)
            observation_, reward, done, info = env.step(action)
            score += reward
            agent.store_transition(observation, action, reward, 
                                    observation_, done)
            agent.learn()
            observation = observation_
            step +=1

            if done:
                if info == "goal":
                    print("Goal")
                    if wins % 10 == 0:
                        env.print_last_five_states()
                    wins+=1
                elif info =="dead":
                    print('Dead')
                    deaths+=1

        scores.append(score)
        eps_history.append(agent.epsilon)
        avg_score = np.mean(scores[-100:])
        print('episode ', i, 'score %.2f' % score,
                'average score %.2f' % avg_score,
                'epsilon %.2f' % agent.epsilon)
        
    print("Wins:", wins)
    print("Deaths:", deaths)
    print("Timeouts:", n_episodes - wins - deaths)
    x = [i+1 for i in range(n_episodes)]
    return wins / n_episodes




if __name__ == "__main__":
    env = Maze(environment_three)
    agent = DDQNAgent(1, 1, 0.00001, 4, [100], 100000, 512, env_name="maze", algo="ddqn")
    train(agent, env, 2500, 150, False, '')
