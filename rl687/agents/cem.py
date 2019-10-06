import numpy as np

from rl687.environments.skeleton import Environment
from .bbo_agent import BBOAgent

from typing import Callable


class CEM(BBOAgent):
    """
    The cross-entropy method (CEM) for policy search is a black box optimization (BBO)
    algorithm. This implementation is based on Stulp and Sigaud (2012). Intuitively,
    CEM starts with a multivariate Gaussian dsitribution over policy parameter vectors.
    This distribution has mean thet and covariance matrix Sigma. It then samples some
    fixed number, K, of policy parameter vectors from this distribution. It evaluates
    these K sampled policies by running each one for N episodes and averaging the
    resulting returns. It then picks the K_e best performing policy parameter
    vectors and fits a multivariate Gaussian to these parameter vectors. The mean and
    covariance matrix for this fit are stored in theta and Sigma and this process
    is repeated.

    Parameters
    ----------
    sigma (float): exploration parameter
    theta (numpy.ndarray): initial mean policy parameter vector
    popSize (int): the population size
    numElite (int): the number of elite policies
    numEpisodes (int): the number of episodes to sample per policy
    evaluationFunction (function): evaluates the provided parameterized policy.
        input: theta_p (numpy.ndarray, a parameterized policy), numEpisodes
        output: the estimated return of the policy
    epsilon (float): small numerical stability parameter
    """
        
    def __init__(self, theta: np.ndarray, sigma: float, popSize: int, numElite: int, numEpisodes: int,
                 evaluationFunction: Callable, epsilon: float = 0.0001):
        self._name = "Cross_Entropy_Method"
        self._theta = theta
        self._initial_theta = theta
        self._sigma = sigma
        self._Sigma = sigma * np.identity(theta.size)
        self._pop_size = popSize
        self._num_elite = numElite
        self._num_episodes = numEpisodes
        self._epsilon = epsilon
        self._evaluate = evaluationFunction

    @property
    def name(self) -> str:
        return self._name

    @property
    def parameters(self) -> np.ndarray:
        return self._theta.flatten()

    def train(self) -> np.ndarray:
        episode_returns = np.zeros(self._pop_size)
        episode_thetas = np.zeros((self._pop_size, self._theta.size))
        for k in range(self._pop_size):
            # self._env.reset()
            theta_k = np.random.multivariate_normal(self.parameters, self._Sigma)
            episode_returns[k] = self._evaluate(theta_k, self._num_episodes)
            episode_thetas[k] = theta_k
            # print(episode_returns[k])

        print("mean population return: ", np.mean(episode_returns))

        elite_index = episode_returns.argsort()[-self._num_elite:]
        # elite_returns = episode_returns[elite_index]
        elite_thetas = episode_thetas[elite_index]
        self._theta = np.mean(elite_thetas, axis=0)
        cov_matrix = self._epsilon * np.identity(self._theta.size)
        temp = elite_thetas - self.parameters
        for i in range(self._num_elite):
            temp2 = temp[i].reshape(self._theta.size, 1)
            cov_matrix += temp2.dot(temp2.T)

        self._Sigma = cov_matrix / (self._epsilon + self._num_elite)
        return self._theta

    def reset(self) -> None:
        self._theta = self._initial_theta
        self._Sigma = self._sigma * np.identity(self._theta.size)
