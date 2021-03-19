import math
import numpy as np
from gym.envs.classic_control.mountain_car import MountainCarEnv
from gym import spaces
from gym.envs.registration import register

class TwoActionMountainCarEnv(MountainCarEnv):
    def __init__(self, goal_velocity=0, seed=None):
        super().__init__(goal_velocity=goal_velocity)
        super().seed(seed)
        #self._orig_action_space = self.action_space
        #self.action_space = spaces.Discrete(2)

    def step(self, action):
        # translate 0 and 1 to 0 and 2 respectively
        #assert self.action_space.contains(action), "%r (%s) invalid" % (action, type(action))
        #action = action + 1
        #assert self._orig_action_space.contains(action), "%r (%s) invalid" % (action, type(action))        

        position, velocity = self.state
        velocity += (action - 1) * self.force + math.cos(3 * position) * (-self.gravity)
        velocity = np.clip(velocity, -self.max_speed, self.max_speed)
        position += velocity
        position = np.clip(position, self.min_position, self.max_position)
        if (position == self.min_position and velocity < 0):
            velocity = 0

        done = bool(
            position >= self.goal_position and velocity >= self.goal_velocity
        )
        reward = -1.0

        self.state = (position, velocity)
        return np.array(self.state), reward, done, {}

#register(
#    id='TwoActionMountainCar-v0',
#    entry_point='experiment.gym_wrappers:TwoActionMountainCarEnv',
#    max_episode_steps=200,
#    reward_threshold=-110.0
#)