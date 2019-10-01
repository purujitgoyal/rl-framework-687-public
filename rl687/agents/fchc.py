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
        self._cov_matrix = sigma * np.identity(theta.size)
        self._num_episodes = numEpisodes
        self._evaluate = evaluationFunction
        self._theta_shape = theta.shape
        self._expected_return = self._evaluate(theta, numEpisodes)

    @property
    def name(self) -> str:
        return self._name

    @property
    def parameters(self) -> np.ndarray:
        return self._theta.flatten()

    def train(self) -> np.ndarray:
        # self._env.reset()
        theta = np.random.multivariate_normal(self.parameters, self._cov_matrix)
        expected_return = self._evaluate(theta, self._num_episodes)
        if expected_return > self._expected_return:
            self._theta = theta
            self._expected_return = expected_return
        # else:
        #     self._expected_return = self._evaluate(self.parameters, self._cov_matrix)

        return self._theta

    def reset(self) -> None:
        self._theta = self._initial_theta
        self._expected_return = self._evaluate(self._theta, self._num_episodes)

