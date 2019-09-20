import numpy as np


class Gridworld():
    """
    The Gridworld as described in the lecture notes of the 687 course material. 
    
    Actions: up (0), down (1), left (2), right (3)
    
    Environment Dynamics: With probability 0.8 the robot moves in the specified
        direction. With probability 0.05 it gets confused and veers to the
        right -- it moves +90 degrees from where it attempted to move, e.g., 
        with probability 0.05, moving up will result in the robot moving right.
        With probability 0.05 it gets confused and veers to the left -- moves
        -90 degrees from where it attempted to move, e.g., with probability 
        0.05, moving right will result in the robot moving down. With 
        probability 0.1 the robot temporarily breaks and does not move at all. 
        If the movement defined by these dynamics would cause the agent to 
        exit the grid (e.g., move up when next to the top wall), then the
        agent does not move. The robot starts in the top left corner, and the 
        process ends in the bottom right corner.
        
    Rewards: -10 for entering the state with water
            +10 for entering the goal state
            0 everywhere else
        
    
    
    """

    def __init__(self, start_state=(0, 0), end_state=(4, 4), shape=(5, 5), obstacles=[(2, 2), (3, 2)],
                 water_states=[(1, 1), (3, 3), (4, 2)], gamma=1):
        self.shape = shape
        self.start_state = start_state
        self.end_state = end_state
        self.water_states = water_states
        self.obstacles = obstacles
        self.action_dict = {"up": 0, "right": 1, "down": 2, "left": 3, "stay": 4}
        self.action_coords = np.array([(-1, 0), (0, 1), (1, 0), (0, -1), (0, 0)])
        self.action_result_dict = {"action_succeeds": 0, "veer_right": 1, "veer_left": 2, "stays": 3}
        self.state = start_state
        self.gamma = gamma

    @staticmethod
    def action_attempted():
        return np.random.choice(4, 1)

    def get_action(self, action_attempted=-1):
        if action_attempted == -1:
            action_attempted = self.action_attempted()

        # print("action_attempted: ", action_attempted[0])
        action_result = np.random.choice(4, 1, p=[0.8, 0.05, 0.05, 0.1])
        # print("action_result: ", action_result)
        if action_result == 0:
            return action_attempted[0]
        elif action_result == 1:
            return (action_attempted[0] + 1) % 4
        elif action_result == 2:
            return (action_attempted[0] - 1) % 4
        elif action_result == 3:
            return self.action_dict["stay"]

    def step(self, action):
        next_state = (self.state[0] + self.action_coords[action][0],
                      self.state[1] + self.action_coords[action][1])

        if next_state in self.obstacles:
            next_state = self.state

        if next_state[0] > self.shape[0] - 1 or next_state[0] < 0:
            next_state = self.state
        elif next_state[1] > self.shape[1] - 1 or next_state[1] < 0:
            next_state = self.state
        # Collect reward
        reward = self.R(next_state, action)

        # Terminate if we reach bottom-right grid corner
        goal = next_state == self.end_state

        # Update state
        self.state = next_state
        return next_state, reward, goal

    def reset(self):
        self.state = self.start_state
        return self.state

    def R(self, state, action):
        """
        reward function
        
        output:
            reward -- the reward resulting in the agent being in a particular state
        """
        if state in self.water_states: #and action != self.action_dict["stay"]:
            return -10
        if state == self.end_state:
            return 10

        return 0
