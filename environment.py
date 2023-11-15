import numpy as np
from collections import deque

ACTIONS = {0: (-1, 0), 1: (1, 0), 2: (0, -1), 3: (0, 1)}

class Maze(object):
    def __init__(self,maze_str):
        self.m_str = maze_str
        self.rows = 0
        self.columns = 0
        self.create_maze_from_string(maze_str)
        self.robot_position = (0,0) # current robot position
        self.steps = 0 # contains num steps robot took
        self.allowed_states = [] # for now, this is none
        self.visited_states = set()
        self.last_five_states = deque(maxlen=5)
        # self.construct_allowed_states() # not implemented yet


    def create_maze_from_string(self, m_str):
        char_to_int = {'0': 0, '1':1, "x":2, "G":3, "A":4}
        def parse_string(s):
            str_list = []
            l = []

            for c in s:
                if c == '\n':
                    str_list.append(l)
                    l = []
                    continue
                l.append(c)

            str_list.pop(0)
            self.rows = len(str_list)
            self.columns = len(str_list[0])
            return str_list
        
        char_list = parse_string(m_str)

        self.maze = np.zeros((self.rows,self.columns))
        for i in range(len(char_list)):
            # print()
            for j in range(len(char_list[0])):
                # print(char_list[i][j])
                self.maze[i,j] = char_to_int[char_list[i][j]]
    
    def reset(self):
        self.create_maze_from_string(self.m_str)
        self.robot_position = (0,0)
        self.visited_states = set()
        return 0
    
    def print_last_five_states(self):
            print("Last Five States of the Maze:")
            for i, state in enumerate(self.last_five_states):
                print(f"State {i + 1}:")
                print(state)
                print()  # Newline for better readability

    def construct_allowed_states(self):
        allowed_states = {}
        for y, row in enumerate(self.maze):
            for x, col in enumerate(row):
                # iterate through all spaces
                if self.maze[(y,x)] != 1:
                    state = (y * self.rows) + x
                    allowed_states[state] = []
                    for action in ACTIONS:
                        if self.is_allowed_move((y,x), action):
                            allowed_states[state].append(action)
        self.allowed_states = allowed_states

    def is_allowed_move(self, state, action):
        y, x = state
        y += ACTIONS[action][0]
        x += ACTIONS[action][1]
        # moving off the board
        if y < 0 or x < 0 or y > self.rows-1 or x > self.columns-1:
            return False
        # moving into start position or empty space
        if self.maze[y, x] == 0 or self.maze[y, x] == 2 or self.maze[y,x] == 3:
            return True
        else:
            return False
        
    def update_maze(self, action):
        y, x = self.robot_position
        self.maze[y, x] = 0  # set the current position to empty
        y += ACTIONS[action][0]
        x += ACTIONS[action][1]
        self.robot_position = (y, x)
        new_state = (y * self.rows) + x

        status = 'normal'
        # Check if new position is a goal or a kill spot
        if self.maze[y, x] == 3:
            self.is_over = True
            status = 'goal'
        elif self.maze[y, x] == 2:
            self.is_over = True
            status = 'dead'
        else:
            status = 'normal'

        self.maze[y, x] = 4  
        self.steps += 1
        return new_state, status
    
    def time_decay_penalty(self):
        # Decrease the reward every 100 steps
        penalty = -1 - (self.steps // 100)
        return penalty

    def give_reward(self, status, new_state):
        # status = self.update_maze()  
        time_penalty = self.time_decay_penalty()
        explore_reward = 0
        if new_state not in self.visited_states:
            explore_reward = 2  # Small positive reward for exploring a new state
            self.visited_states.add(new_state)


        if status == 'goal':
            return 100 
        elif status == 'dead':
            return -50
        else:
            return -1
        
    def step(self, action):
        new_state, status = self.update_maze(action)
        self.last_five_states.append(np.copy(self.maze))
        reward = self.give_reward(status, new_state)
        is_over = self.is_game_over(status)
        return new_state, reward, status, is_over

    def is_game_over(self, status):
        if status == 'dead' or status == 'goal':
            return True
        else:
            False
        
        
    def get_state_and_reward(self):
        return self.robot_position, self.give_reward()

        



