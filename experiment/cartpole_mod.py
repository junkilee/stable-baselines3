import gym

from stable_baselines3 import PPO
import experiment.gym_wrappers as gym_wrappers

import numpy as np
import matplotlib.pyplot as plt

import random
from datetime import datetime

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("timesteps", help="number of training time steps", type=int, default=10000)
parser.add_argument("--seed", help="random seed for training", type=int)
parser.add_argument("--uniform", help="whether to use uniform start states from -2.4 to 2.4", action="store_true")
args = parser.parse_args()

seed = None
if args.seed is None:
  random.seed(datetime.now())
  seed = random.randint(0, 1000)
else:
  seed = args.seed
print("seed : {}".format(seed))

name_str = ""
env = None
if args.uniform:
  name_str += "_uniform"
  env = gym_wrappers.make("cartpole", "uniform", "0", 
                          start_state_mode =gym_wrappers.StartStateMode.UNIFORM,
                          start_states=[-2.4,2.4], seed=seed)
else:
  env = gym_wrappers.make("cartpole", "uniform", "0", 
                        start_state_mode =gym_wrappers.StartStateMode.DESIGNATED_POSITIONS,
                        start_states=[0.0], seed=seed)

num_models = 5

models = [PPO('MlpPolicy', env, verbose=1, seed=seed+i*7) for i in range(num_models)]

for i in range(num_models):
  models[i].learn(total_timesteps=args.timesteps)  

name_str += "_" + str(args.timesteps//1000) + "k"

env = gym_wrappers.make("cartpole", "uniform", "0", 
                        start_state_mode =gym_wrappers.StartStateMode.UNIFORM,
                        start_states=[-2.4,2.4], seed=seed+seed)

num_x_datapoints = 20
num_y_datapoints = 10

accu_rewards = np.zeros((num_x_datapoints, num_y_datapoints * num_models))
idx = 0
xrange = np.linspace(-2.4, 2.4, num_x_datapoints)
for x in xrange:
  for k in range(num_models):      
    for j in range(num_y_datapoints):    
      obs = env.reset()
      obs[0] = x + np.random.uniform(low=-0.05, high=0.05)
      env.manual_set(obs)
      for i in range(1000):
        action, _states = models[k].predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        accu_rewards[idx, k*num_y_datapoints + j] += reward

        #env.render()
        if done:        
          break
  idx += 1

avg_rewards = np.average(accu_rewards, axis=1)
std_err = np.std(accu_rewards, axis=1) / np.sqrt(num_y_datapoints)

env.close()
plt.errorbar(xrange, avg_rewards, std_err)
plt.ylim((-50, 1100))

plt.savefig("results{}.pdf".format(name_str))
plt.savefig("results{}.png".format(name_str))
plt.close()