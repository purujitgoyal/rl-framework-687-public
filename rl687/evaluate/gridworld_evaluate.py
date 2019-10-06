import numpy as np
import matplotlib.pyplot as plt

from rl687.environments.gridworld import Gridworld
from rl687.policies.tabular_softmax import TabularSoftmax


class GridworldEvaluate:

    def __init__(self, numStates: int, numActions: int):
        self._returns_iteration = []
        self._num_states = numStates
        self._num_actions = numActions
        self._gridworld = Gridworld()
        plt.rcParams['figure.figsize'] = (10.0, 8.0)
        plt.rcParams['image.interpolation'] = 'nearest'
        plt.rcParams['image.cmap'] = 'gray'

    def __call__(self, theta: np.ndarray, num_episodes: int, to_store: bool = True):
        returns = np.zeros(num_episodes)
        for episode in range(num_episodes):
            self._gridworld.reset()
            step = 0
            g = 0
            tabular_policy = TabularSoftmax(numStates=self._num_states, numActions=self._num_actions)
            tabular_policy.parameters = theta
            while not self._gridworld.isEnd:
                s, r, e = self._gridworld.step(tabular_policy.sampleAction(self._gridworld.state))
                g += (self._gridworld.gamma ** step) * r
                step += 1
                if step > 200:
                    g += -50
                    break

            returns[episode] = g
            if to_store:
                self._returns_iteration.append(g)
            # print(g)
        # print("Average: {}\nStandard Deviation: {}\nMin: {}\nMax: {}".format( \
        #     np.mean(returns), np.std(returns), np.min(returns), np.max(returns)))
        return np.mean(returns)

    def returns(self):
        return self._returns_iteration

    def reset(self):
        self._returns_iteration = []

    def plot(self, num_trials: int = 1):
        returns_iter_array = np.asarray(self._returns_iteration).reshape(num_trials, -1)
        mean = np.mean(returns_iter_array, axis=0)
        std = np.std(returns_iter_array, axis=0)
        plt.errorbar(np.arange(returns_iter_array.shape[1]), mean, yerr=std, fmt='o', ecolor='gray')
        plt.xlabel('n_episodes')
        plt.ylabel('mean_reward')
        # plt.title('Quantile Distribution')
        # plt.legend()
        plt.show()
