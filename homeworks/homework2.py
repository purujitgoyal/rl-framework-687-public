import numpy as np

from rl687.agents.cem import CEM
from rl687.environments.gridworld import Gridworld
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

    def evaluate(theta: np.ndarray, num_episodes: int):
        returns = np.zeros(num_episodes)
        for episode in range(num_episodes):
            gridworld.reset()
            step = 0
            g = 0
            tabular_policy = TabularSoftmax(numStates=25, numActions=4, theta=theta.reshape(25, 4))
            while not gridworld.isEnd:
                s, r, e = gridworld.step(tabular_policy.sampleAction(gridworld.state))
                g += (gridworld.gamma ** step) * r
                step += 1
            returns[episode] = g
            print(g)
        # print("Average: {}\nStandard Deviation: {}\nMin: {}\nMax: {}".format( \
        #     np.mean(returns), np.std(returns), np.min(returns), np.max(returns)))
        return np.mean(returns)

    cem = CEM(np.zeros(100), 0.1, 10, 5, 3, evaluate)
    cem.train()


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

    # TODO
    pass


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
    problem1()

    # TODO
    pass


if __name__ == "__main__":
    main()
