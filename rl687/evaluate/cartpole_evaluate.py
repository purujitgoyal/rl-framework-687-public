import itertools

import numpy as np
import matplotlib.pyplot as plt

from rl687.environments.cartpole import Cartpole
from rl687.policies.tabular_softmax import TabularSoftmax


class CartpoleEvaluate:

    def __init__(self, numStateVariables: int, numActions: int, numFeaturesSize: int):
        self._returns_iteration = []
        self._num_states_variables = numStateVariables
        self._num_actions = numActions
        self._num_features_size = numFeaturesSize
        self._cartpole = Cartpole()
        self._state_feature_vectors = np.asarray(
            list(itertools.product(np.arange(self._num_features_size), repeat=self._num_states_variables)))
        self._x_min = -3.0
        self._x_max = 3.0
        self._v_min = -5.0
        self._v_max = 5.0
        self._theta_min = -np.pi / 12.0
        self._theta_max = np.pi / 12.0
        self._dtheta_min = -np.pi
        self._dtheta_max = np.pi
        plt.rcParams['figure.figsize'] = (10.0, 8.0)
        plt.rcParams['image.interpolation'] = 'nearest'
        plt.rcParams['image.cmap'] = 'gray'

    def __call__(self, theta_action: np.ndarray, num_episodes: int, to_store: bool = True):
        returns = np.zeros(num_episodes)
        for episode in range(num_episodes):
            self._cartpole.reset()
            step = 0
            g = 0

            while not self._cartpole.isEnd:
                state = self.normalize_state(self._cartpole.state)
                phi_state = np.cos(np.pi*self._state_feature_vectors.dot(state))
                theta = theta_action.reshape(phi_state.shape[0], -1).T.dot(phi_state)
                tabular_policy = TabularSoftmax(numStates=1, numActions=self._num_actions)
                tabular_policy.parameters = theta
                s, r, e = self._cartpole.step(tabular_policy.sampleAction(0))
                g += (self._cartpole.gamma ** step) * r
                step += 1

            returns[episode] = g
            if to_store:
                self._returns_iteration.append(g)
            # print(g)
        # print("Average: {}\nStandard Deviation: {}\nMin: {}\nMax: {}".format( \
        #     np.mean(returns), np.std(returns), np.min(returns), np.max(returns)))
        return np.mean(returns)

    def returns(self):
        return self._returns_iteration

    def normalize_state(self, state: np.ndarray):
        state -= np.array([self._x_min, self._v_min, self._theta_min, self._dtheta_min])
        state /= np.array([self._x_max - self._x_min, self._v_max - self._v_min, self._theta_max - self._theta_min,
                           self._dtheta_max - self._dtheta_min])
        return state

    def reset(self):
        self._returns_iteration = []

    def plot(self, num_trials: int = 1):
        returns_iter_array = np.asarray(self._returns_iteration).reshape(num_trials, -1)
        mean = np.mean(returns_iter_array, axis=0)
        std = np.std(returns_iter_array, axis=0)
        plt.errorbar(np.arange(returns_iter_array.shape[1]), mean, yerr=std, fmt='o', ecolor='gray')
        plt.xlabel('n_episodes')
        plt.ylabel('mean_reward')

        plt.show()
