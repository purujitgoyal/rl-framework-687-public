import numpy as np
from typing import Tuple
from .skeleton import Environment


class Cartpole(Environment):
    """
    The cart-pole environment as described in the 687 course material. This
    domain is modeled as a pole balancing on a cart. The agent must learn to
    move the cart forwards and backwards to keep the pole from falling.

    Actions: left (0) and right (1)
    Reward: 1 always

    Environment Dynamics: See the work of Florian 2007
    (Correct equations for the dynamics of the cart-pole system) for the
    observation of the correct dynamics.
    """

    def __init__(self):
        self._name = "Cartpole"
        self._action = None
        self._reward = 0.
        self._isEnd = False
        self._gamma = 1.

        # define the state # NOTE: you must use these variable names
        self._x = 0.  # horizontal position of cart
        self._v = 0.  # horizontal velocity of the cart
        self._theta = 0.  # angle of the pole
        self._dtheta = 0.  # angular velocity of the pole

        # dynamics
        self._g = 9.8  # gravitational acceleration (m/s^2)
        self._mp = 0.1  # pole mass
        self._mc = 1.0  # cart mass
        self._l = 0.5  # (1/2) * pole length
        self._dt = 0.02  # timestep
        self._t = 0.0  # total time elapsed  NOTE: USE must use this variable
        self._maxF = 10.0  # max force on cart
        self._failAngle = np.pi/12.0

    @property
    def name(self) -> str:
        return self._name

    @property
    def reward(self) -> float:
        return self._reward

    @property
    def gamma(self) -> float:
        return self._gamma

    @property
    def action(self) -> int:
        return self._action

    @property
    def isEnd(self) -> bool:
        return self._isEnd

    @property
    def state(self) -> np.ndarray:
        return np.array([self._x, self._v, self._theta, self._dtheta])

    def nextState(self, state: np.ndarray, action: int) -> np.ndarray:
        """
        Compute the next state of the pendulum using the euler approximation to the dynamics
        """
        if self.isEnd:
            return state

        force = self._maxF
        if action == 0:
            force *= -1

        x_dot = state[1]
        theta_dot = state[3]

        cos_theta = np.cos(state[2])
        sin_theta = np.sin(state[2])
        total_mass = self._mp + self._mc

        temp1 = (force + self._mp*self._l*(state[3]**2)*sin_theta)/total_mass
        temp2 = 4.0/3.0 - (self._mp*(cos_theta**2)/total_mass)

        dtheta_dot = (self._g*sin_theta - cos_theta*temp1)/(self._l*temp2)
        v_dot = (force + self._mp*self._l*((theta_dot**2)*sin_theta - dtheta_dot*cos_theta))/total_mass

        next_state = state + self._dt*np.array([x_dot, v_dot, theta_dot, dtheta_dot])
        return next_state

    def R(self, state: np.ndarray, action: int, nextState: np.ndarray) -> float:
        return 0 if (np.all(state == nextState) and self.isEnd) else 1

    def step(self, action: int) -> Tuple[np.ndarray, float, bool]:
        """
        takes one step in the environment and returns the next state, reward, and if it is in the terminal state
        """
        next_state = self.nextState(self.state, action)
        self._action = action
        self._reward = self.R(self.state, action, next_state)
        self._t += self._dt
        self._x = next_state[0]
        self._v = next_state[1]
        self._theta = next_state[2]
        self._dtheta = next_state[3]
        self._isEnd = self.terminal()

        return next_state, self.reward, self.isEnd

    def reset(self) -> None:
        """
        resets the state of the environment to the initial configuration
        """
        self._x = 0.
        self._v = 0.
        self._theta = 0.
        self._dtheta = 0.
        self._reward = 0.
        self._t = 0.
        self._isEnd = False
        self._action = None

    def terminal(self) -> bool:
        """
        The episode is at an end if:
            time is greater that 20 seconds
            pole falls |theta| > (pi/12.0)
            cart hits the sides |x| >= 3
        """
        if (np.abs(self._theta) > self._failAngle) or np.abs(self._x) >= 3 or self._t > 20:
            return True

        return False
