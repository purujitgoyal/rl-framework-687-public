import numpy as np

from rl687.environments.skeleton import Environment
from .bbo_agent import BBOAgent

from typing import Callable


class GA(BBOAgent):
    """
    A canonical Genetic Algorithm (GA) for policy search is a black box 
    optimization (BBO) algorithm. 
    
    Parameters
    ----------
    populationSize (int): the number of individuals per generation
    numEpisodes (int): the number of episodes to sample per policy         
    evaluationFunction (function): evaluates a parameterized policy
        input: a parameterized policy theta, numEpisodes
        output: the estimated return of the policy            
    initPopulationFunction (function): creates the first generation of
                    individuals to be used for the GA
        input: populationSize (int)
        output: a numpy matrix of size (N x M) where N is the number of 
                individuals in the population and M is the number of 
                parameters (size of the parameter vector)
    numElite (int): the number of top individuals from the current generation
                    to be copied (unmodified) to the next generation
    
    """

    def __init__(self, populationSize: int, evaluationFunction: Callable,
                 initPopulationFunction: Callable, numElite: int = 1, numEpisodes: int = 10, ):
        self._name = "Genetic_Algorithm"
        self._init_population = initPopulationFunction
        self._evaluate = evaluationFunction
        self._num_elite = numElite
        self._num_episodes = numEpisodes
        self._num_parents = 5
        self._num_children = populationSize - numElite
        self._population_size = populationSize
        self._population = self._init_population(self._population_size)
        self._alpha = 2.5
        self._parameters = np.zeros(self._population.shape[1], 1)

    @property
    def name(self) -> str:
        return self._name

    @property
    def parameters(self) -> np.ndarray:
        return self._parameters

    def _mutate(self, parent: np.ndarray) -> np.ndarray:
        """
        Perform a mutation operation to create a child for the next generation.
        The parent must remain unmodified. 
        
        output:
            child -- a mutated copy of the parent
        """
        epsilon = np.random.standard_normal(parent.size)
        return parent + self._alpha * epsilon

    def train(self) -> np.ndarray:
        episode_returns = np.zeros(self._population.shape[0])
        episode_thetas = np.zeros((self._population.shape[0], self._population.shape[1]))
        for k in range(self._population.shape[0]):
            theta_k = self._population[k, :]
            episode_returns[k] = self._evaluate(theta_k, self._num_episodes)
            episode_thetas[k] = theta_k

        sorted_returns_index = episode_returns.argsort()
        elite_index = sorted_returns_index[-self._num_elite:]
        elite_thetas = episode_thetas[elite_index]
        parents_index = sorted_returns_index[-self._num_parents:]
        parents_thetas = episode_thetas[parents_index]
        child_thetas = np.zeros((self._num_children, self._population.shape[1]))
        random_parents = np.random.choice(parents_thetas.shape[0], self._num_children)
        for i in range(self._num_children):
            child_thetas[i] = self._mutate(parents_thetas[random_parents[i]])

        self._population = np.append(elite_thetas, child_thetas, axis=0)
        self._parameters = np.mean(self._population, axis=0)

        return self._population

    def reset(self) -> None:
        self._population = self._init_population(self._population_size)
        self._parameters = np.zeros(self._population.shape[1], 1)
