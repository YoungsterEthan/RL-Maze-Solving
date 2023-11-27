import matplotlib.pyplot as plt
import numpy as np
import torch
from agent import DDQNAgent
from environment import Maze


def plotLearning(x, scores, epsilons, filename, lines=None):
    fig=plt.figure()
    ax=fig.add_subplot(111, label="1")
    ax2=fig.add_subplot(111, label="2", frame_on=False)

    ax.plot(x, epsilons, color="C0")
    ax.set_xlabel("Game", color="C0")
    ax.set_ylabel("Epsilon", color="C0")
    ax.tick_params(axis='x', colors="C0")
    ax.tick_params(axis='y', colors="C0")

    N = len(scores)
    running_avg = np.empty(N)
    for t in range(N):
        running_avg[t] = np.mean(scores[max(0, t-20):(t+1)])

    ax2.scatter(x, running_avg, color="C1")
    #ax2.xaxis.tick_top()
    ax2.axes.get_xaxis().set_visible(False)
    ax2.yaxis.tick_right()
    #ax2.set_xlabel('x label 2', color="C1")
    ax2.set_ylabel('Score', color="C1")
    #ax2.xaxis.set_label_position('top')
    ax2.yaxis.set_label_position('right')
    #ax2.tick_params(axis='x', colors="C1")
    ax2.tick_params(axis='y', colors="C1")

    if lines is not None:
        for line in lines:
            plt.axvline(x=line)

    plt.savefig(filename)

def generate_policy(agent:DDQNAgent, env:Maze, num_rows, num_cols):
    policy = np.zeros((num_rows, num_cols), dtype=int)
    env.reset()
    for row in range(num_rows):
        for col in range(num_cols):
            if env.maze[row, col] == 1 or env.maze[row, col] == 3:
                continue
            env.maze[row, col] = 4
            state = np.copy(env.maze).flatten()
            best_action = agent.choose_action(state)

            policy[row, col] = best_action
            env.maze[row, col] = 0
    return policy
action_to_int = {'L': 0, 'D': 1, 'R': 2, 'U': 3}

def visualize_path(policy, start, goal):
    current = start
    path = [start]
    while current != goal:
        action = policy[current]
        # Update current state based on action
        # Example for a grid: 
        move = None
        if action == 0: 
            move = (0,-1)
        if action == 1: 
            move = (1,0)
        if action == 2:
            move = (0,1)
        if action == 3:
            move = (-1,0)
        current = tuple(np.array(current) + np.array(move))
        path.append(current)
        if len(path) > policy.size:  # Prevent infinite loop
            break

    # Plotting
    print(path)
    fig, ax = plt.subplots()
    ax.matshow(np.zeros(policy.shape), cmap='gray')

    for (y, x) in path:
        ax.text(x, y, '>', va='center', ha='center')

    ax.text(start[1], start[0], 'S', va='center', ha='center', color='green')
    ax.text(goal[1], goal[0], 'G', va='center', ha='center', color='red')

    plt.show()

def show(qmaze: Maze):
    plt.grid('on')
    nrows = qmaze.rows
    ncols = qmaze.columns
    ax = plt.gca()
    ax.set_xticks(np.arange(0.5, nrows, 1))
    ax.set_yticks(np.arange(0.5, ncols, 1))
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    canvas = np.copy(qmaze.maze)
    for row,col in qmaze.visited_states:
        canvas[row,col] = 0.6
    rrow, col = qmaze.robot_position
    canvas[row, col] = 0.3   # rat cell
    img = plt.imshow(canvas, interpolation='none', cmap='gray')
    plt.show()
    return img