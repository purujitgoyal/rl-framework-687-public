import numpy as np
import matplotlib.pyplot as plt
from rl687.environments.gridworld import Gridworld

plt.rcParams['figure.figsize'] = (10.0, 8.0)  # set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'


def problem_a(n_episodes=10000):  # problem: 1
    """
    Have the agent uniformly randomly select actions. Run 10,000 episodes.
    Report the mean, standard deviation, maximum, and minimum of the observed 
    discounted returns.
    """
    gridworld = Gridworld(gamma=0.9)
    # n_episodes = 10000
    rewards = np.zeros(n_episodes)
    np.random.seed(n_episodes)
    for episode in range(n_episodes):
        # print(episode)
        gridworld.reset()  # starting state
        t = 0
        while True:
            action = gridworld.get_action()  # get action
            # print("action: ", action)
            next_state, reward, done = gridworld.step(action)  # evolve state by action
            rewards[episode] += np.power(gridworld.gamma, t) * reward
            if done:
                break
            t += 1
            # print(next_state)

    return np.sort(rewards)


def problem_b(n_episodes=10000):  # problem: 3
    """
    Run the optimal policy that you found for 10,000 episodes. Report the
    mean, standard deviation, maximum, and minimum of the observed 
    discounted returns
    """
    gridworld = Gridworld(gamma=0.9)
    # n_episodes = 10000
    rewards = np.zeros(n_episodes)
    optimal_policy = np.array([[1, 1, 1, 1, 2],
                               [0, 1, 1, 1, 2],
                               [0, 2, -1, 1, 2],
                               [0, 3, -1, 0, 2],
                               [0, 3, 1, 1, -1]])

    np.random.seed(n_episodes)
    for episode in range(n_episodes):
        # print(episode)
        gridworld.reset()  # starting state
        t = 0
        while True:
            optimal_action = np.array(optimal_policy[gridworld.state[0]][gridworld.state[1]]).reshape(1, )
            action = gridworld.get_action(optimal_action)
            next_state, reward, done = gridworld.step(action)  # evolve state by action
            rewards[episode] += np.power(gridworld.gamma, t) * reward
            if done:
                break
            t += 1
            # print(next_state)

    return np.sort(rewards)


def problem_e():  # problem: 5
    """
    Using simulations, empirically estimate the probability that
    S19 = 21 given that S8 = 18 (the state above the goal) when running the
    uniform random policy
    Event A: P(S19=(4,2) and S8=(3,4)
    Event B: P(S8=(3,4))
    :return
        P(A)/P(B)
    """
    gridworld = Gridworld(start_state=(3, 4), gamma=0.9)
    n_b = 0
    n_ab = 0
    episode = 0
    while n_b < 10000:
        print(episode)
        gridworld.reset()  # starting state
        t = 0
        while t < 11:
            action = gridworld.get_action()  # get action
            next_state, reward, done = gridworld.step(action)  # evolve state by action
            # rewards[episode] += np.power(gridworld.gamma, t) * reward
            if done:
                break
            t += 1
            # print(next_state)
        episode += 1
        n_b += 1
        if t == 11 and gridworld.state == (4, 2):
            n_ab += 1

    return n_ab, n_b, (n_ab / n_b)


def problem_d():  # problem: 4
    """
    Plot the distribution of returns for both the random policy and the optimal
    policy using 10,000 trials each.
    """
    rewards_uniform = problem_a(n_episodes=10000)
    rewards_optimal = problem_b(n_episodes=10000)
    steps = np.linspace(0, 1, 1000)
    # print(rewards_optimal.shape)
    quantile_uniform = np.quantile(rewards_uniform, steps)
    quantile_optimal = np.quantile(rewards_optimal, steps)
    plt.scatter(steps, quantile_optimal, color='r', label='quantile_optimal')
    plt.scatter(steps, quantile_uniform, color='b', label='quantile_uniform')
    plt.xlabel('CDF Value')
    plt.ylabel('Reward value')
    plt.title('Quantile Distribution')
    plt.legend()
    plt.show()


def describe(a: np.array):
    print("Mean: ", np.mean(a))
    print("Standard Deviation: ", np.std(a))
    print("Max: ", np.max(a))
    print("Min: ", np.min(a))


def main(problem=-1):
    if problem == 1:
        describe(problem_a())
    elif problem == 3:
        describe(problem_b())
    elif problem == 4:
        problem_d()
    elif problem == 5:
        print(problem_e())
    else:
        print("Invalid Problem Number")


if __name__ == '__main__':
    main(4)
