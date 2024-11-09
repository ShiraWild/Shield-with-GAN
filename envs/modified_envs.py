import numpy as np
import gymnasium
from envs.cartpole import CartPoleEnv

class CartPoleWithCost(CartPoleEnv):
    def __init__(self, safe_limit_x, safe_limit_theta):
        super().__init__()
        self.safe_limit_x = safe_limit_x
        self.safe_limit_theta = safe_limit_theta

    def get_cost(self, state):
        theta = state[2]
        x = state[0]
        safety_x_threshold =  self.x_threshold - self.safe_limit_x
        safety_theta_threshold_radians = self.theta_threshold_radians -  self.safe_limit_theta
        safe_score = bool(
            x < -safety_x_threshold
            or x > safety_x_threshold
            or theta < -safety_theta_threshold_radians
            or theta > safety_theta_threshold_radians
        )
        # returns 0 if safe and 1 if not safe
        return safe_score

    def step(self, action):
        state, reward, done, _, info = super().step(action)
        info["cost"] = self.get_cost(state)
        return state, reward, done, info

