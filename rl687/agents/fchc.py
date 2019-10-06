import numpy as np

from rl687.environments.skeleton import Environment
from .bbo_agent import BBOAgent

from typing import Callable


class FCHC(BBOAgent):
    """
    First-choice hill-climbing (FCHC) for policy search is a black box optimization (BBO)
    algorithm. This implementation is a variant of Russell et al., 2003. It has 
    remarkably fewer hyperparameters than CEM, which makes it easier to apply. 
    
    Parameters
    ----------
    sigma (float): exploration parameter 
    theta (np.ndarray): initial mean policy parameter vector
    numEpisodes (int): the number of episodes to sample per policy
    evaluationFunction (function): evaluates a provided policy.
        input: policy (np.ndarray: a parameterized policy), numEpisodes
        output: the estimated return of the policy 
    """

    def __init__(self, theta: np.ndarray, sigma: float, evaluationFunction: Callable, numEpisodes: int = 10):
        self._name = "First_Choice_Hill_Climbing"
        self._theta = theta
        self._initial_theta = theta
        self._sigma = sigma
        self._initial_sigma = sigma
        self._cov_matrix = sigma * np.identity(theta.size)
        self._num_episodes = numEpisodes
        self._evaluate = evaluationFunction
        self._theta_shape = theta.shape
        self._expected_return = self._evaluate(theta, numEpisodes)
        self._counter = 0

    @property
    def name(self) -> str:
        return self._name

    @property
    def parameters(self) -> np.ndarray:
        return self._theta.flatten()

    def train(self) -> np.ndarray:
        # self._env.reset()
        if self._counter > 20:
            print("Updating max return yet")
            self._expected_return = self._evaluate(self.parameters, self._num_episodes, False)
            self._counter = 0
            # self._sigma *= 0.99
            # self._cov_matrix = self._sigma * np.identity(self._theta.size)
            # print(self._sigma)

        theta = np.random.multivariate_normal(self.parameters, self._cov_matrix)
        expected_return = self._evaluate(theta, self._num_episodes)
        if expected_return > self._expected_return:
            self._theta = theta
            self._expected_return = expected_return
            self._counter = 0
        else:
            self._counter += 1

        print(expected_return)
        print("max return yet: ", self._expected_return)
        return self._theta

    def reset(self) -> None:
        self._theta = self._initial_theta
        self._expected_return = self._evaluate(self._theta, self._num_episodes)
        self._sigma = self._initial_sigma
        self._cov_matrix = self._sigma * np.identity(self._theta.size)
        self._counter = 0


