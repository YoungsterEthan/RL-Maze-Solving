import numpy as np
from agent import Agent
from environment import Maze
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import seaborn as sns
from utils import plotLearning

ACTIONS = {0: (-1, 0), 1: (1, 0), 2: (0, -1), 3: (0, 1)}
action_to_int = {'U': 0, 'D': 1, 'L': 2, 'R': 3}

# environment = """
# A00000000x
# 0000000000
# 0000000000
# 0011111000
# 0010G01000
# 0010001000
# 0000000000
# 0000000000
# 0000000000
# x00000000x
# """

environment = """
A00
000
x0G
"""


def train(agent, env, n_episodes, n_steps, plot, filename):
    deaths = 0
    wins = 0
    scores = []
    eps_history = []
    for i in range(n_episodes):
        score = 0
        step = 0
        done = False
        observation = env.reset()
        while not done and step < n_steps:
            print(step)
            action = agent.choose_action(observation)
            while not env.is_allowed_move(observation, action):
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
                    wins+=1
                elif info =="dead":
                    deaths+=1
        scores.append(score)
        # if score > 0:
        #     env.print_last_five_states()
        eps_history.append(agent.epsilon)

        avg_score = np.mean(scores[-100:])

        print('episode ', i, 'score %.2f' % score,
                'average score %.2f' % avg_score,
                'epsilon %.2f' % agent.epsilon)
        
    print("Wins:", wins)
    print("Deaths:", deaths)
    print("Timeouts:", n_episodes - wins - deaths)
    x = [i+1 for i in range(n_episodes)]
    if plot:
        plotLearning(x, scores, eps_history, filename)

    return wins / n_episodes
def evaluate_parameters(gamma, lr, batch_size, max_mem_size, eps_end, eps_dec):
    win_pct_list = []
    plot = False
    plot_name = f"eval_gamma_{gamma}_lr_{lr}_batch_{batch_size}_mem_{max_mem_size}_epsend_{eps_end}_epsdec_{eps_dec}.png"
    for i in range(5):
        if i == 0:
            plot = True
        else:
            plot = False
        env = Maze(environment)
        agent = Agent(gamma=gamma, epsilon=1.0, lr=lr, input_dims=[1], batch_size=batch_size, n_actions=4,
                    max_mem_size=max_mem_size, eps_end=eps_end, eps_dec=eps_end)
        win_pct = train(agent, env, 500, 500, plot, plot_name)
        win_pct_list.append(win_pct)


    return sum(win_pct_list) / len(win_pct_list)


if __name__ == "__main__":
    env = Maze(environment)
    agent = Agent(1, 1, 0.001, [1], 1, 4, 1, eps_dec=0.0005, eps_end=0.05)
    train(agent, env, 1500, 25, False, '')




    # eps_decays = [5e-3,5e-4,5e-5]
    # max_mem_sizes = [10, 100, 500, 1000, 2000, 10000, 100000]
    # batch_sizes = [1,4,16,32,64,128]
    # # learning_rates = [0.1, 0.01, 0.001, 0.0001]
    # # eps_ends = [0.01, 0.05, 0.1]

    # for mem_size in max_mem_sizes:
    #     for batch_size in batch_sizes:
    #         avg = evaluate_parameters(1,0.001, batch_size, mem_size, 0.05, 5e-4)
    #         print(f"WIN PCT for eval_{batch_size}_mem_{mem_size}: {avg}")

















    
#   # We iterate over episodes
#     scores, eps_history = [], []
#     for i in range(n_training_episodes):
#         score = 0
#         step = 0
#         done = False
#         observation = env.reset()
#         while not done and step < max_steps:
#             action = agent.choose_action(observation)
#             while not env.is_allowed_move(observation, action):
#                 action = agent.choose_action(observation)
#             observation_, reward, done, info = env.step(action)
#             score += reward
#             agent.store_transition(observation, action, reward, 
#                                     observation_, done)
#             agent.learn()
#             observation = observation_
#             step +=1

#             if done:
#                 if info == "goal":
#                     wins+=1
#                 elif info =="dead":
#                     deaths+=1
#         scores.append(score)
#         # if score > 0:
#         #     env.print_last_five_states()
#         eps_history.append(agent.epsilon)

#         avg_score = np.mean(scores[-100:])

#         print('episode ', i, 'score %.2f' % score,
#                 'average score %.2f' % avg_score,
#                 'epsilon %.2f' % agent.epsilon)
#     print("Wins:", wins)
#     print("Deaths:", deaths)
#     print("Timeouts:", n_training_episodes - wins - deaths)

#     x = [i+1 for i in range(n_training_episodes)]
#     filename = 'maze_runner.png'
#     plotLearning(x, scores, eps_history, filename)


    