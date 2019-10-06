import numpy as np

from rl687.agents.cem import CEM
from rl687.agents.fchc import FCHC
from rl687.agents.ga import GA
from rl687.environments.cartpole import Cartpole
from rl687.environments.gridworld import Gridworld
from rl687.evaluate.cartpole_evaluate import CartpoleEvaluate
from rl687.evaluate.gridworld_evaluate import GridworldEvaluate
from rl687.policies.tabular_softmax import TabularSoftmax


def problem1():
    """
    Apply the CEM algorithm to the More-Watery 687-Gridworld. Use a tabular 
    softmax policy. Search the space of hyperparameters for hyperparameters 
    that work well. Report how you searched the hyperparameters, 
    what hyperparameters you found worked best, and present a learning curve
    plot using these hyperparameters, as described in class. This plot may be 
    over any number of episodes, but should show convergence to a nearly 
    optimal policy. The plot should average over at least 500 trials and 
    should include standard error or standard deviation error bars. Say which 
    error bar variant you used. 
    """
    gridworld = Gridworld()

    def init_population(population_size: int):
        return np.random.standard_normal((population_size, 100))

    gridworld_evaluate = GridworldEvaluate(numStates=25, numActions=4)

    # cem = CEM(theta=np.zeros(100), sigma=0.2, popSize=50, numElite=12, numEpisodes=50,
    #           evaluationFunction=gridworld_evaluate, epsilon=1.5)
    # gridworld_evaluate.reset()
    # for t in range(5):
    #     print("Trial: ", t)
    #     cem.reset()
    #     for i in range(100):
    #         print("iteration: ", i)
    #         cem.train()
    #     print(len(gridworld_evaluate.returns()))
    #
    # gridworld_evaluate.plot(5)
    #
    fchc = FCHC(theta=np.zeros(100), sigma=0.35, evaluationFunction=gridworld_evaluate, numEpisodes=200)
    gridworld_evaluate.reset()
    for t in range(50):
        print("Trial: ", t)
        fchc.reset()
        for i in range(500):
            print("iteration: ", i)
            fchc.train()

    gridworld_evaluate.plot(50)

    # print(len(gridworld_evaluate.returns()))
    #
    # print("final expected return")
    # print(evaluate(fchc.parameters, 200))
    #
    # ga = GA(populationSize=50, evaluationFunction=gridworld_evaluate, initPopulationFunction=init_population, numElite=15,
    #         numEpisodes=50, numParents=20, alpha=1.5)
    # for i in range(100):
    #     print("iteration: ", i)
    #     ga.train()

    # print(len(gridworld_evaluate.returns()))

    # print("final expected return")
    # print(evaluate(cem.parameters, 200))


def problem2():
    """
    Repeat the previous question, but using first-choice hill-climbing on the 
    More-Watery 687-Gridworld domain. Report the same quantities.
    """

    # TODO
    pass


def problem3():
    """
    Repeat the previous question, but using the GA (as described earlier in 
    this assignment) on the More-Watery 687-Gridworld domain. Report the same 
    quantities.
    """

    # TODO
    pass


def problem4():
    """
    Repeat the previous question, but using the cross-entropy method on the 
    cart-pole domain. Notice that the state is not discrete, and so you cannot 
    directly apply a tabular softmax policy. It is up to you to create a 
    representation for the policy for this problem. Consider using the softmax 
    action selection using linear function approximation as described in the notes. 
    Report the same quantities, as well as how you parameterized the policy. 
    
    """
    def init_population(population_size: int):
        return np.random.standard_normal((population_size, 32))
    cartpole_evaluate = CartpoleEvaluate(numStateVariables=4, numActions=2, numFeaturesSize=2)
    # cem = CEM(theta=np.zeros(32), sigma=3.5, popSize=25, numElite=6, numEpisodes=20,
    #           evaluationFunction=cartpole_evaluate, epsilon=2)
    ga = GA(populationSize=25, evaluationFunction=cartpole_evaluate, initPopulationFunction=init_population, numElite=6, numEpisodes=20, numParents=6, alpha=2.5)
    cartpole_evaluate.reset()
    for t in range(1):
        print("Trial: ", t)
        ga.reset()
        for i in range(50):
            print("iteration: ", i)
            ga.train()

    print("final expected return")
    print(cartpole_evaluate(ga.parameters, 200, False))

    cartpole_evaluate.plot(4)


def problem5():
    """
    Repeat the previous question, but using first-choice hill-climbing (as 
    described in class) on the cart-pole domain. Report the same quantities 
    and how the policy was parameterized. 
    
    """
    # TODO
    pass


def problem6():
    """
    Repeat the previous question, but using the GA (as described earlier in 
    this homework) on the cart-pole domain. Report the same quantities and how
    the policy was parameterized. 
    """

    # TODO
    pass


def main():
    print("hello, world")
    problem4()
    # cartpole = Cartpole()
    # for i in range(11):
    #     print("step: ", i)
    #     print(cartpole.step(0))
    # TODO
    pass


if __name__ == "__main__":
    main()
