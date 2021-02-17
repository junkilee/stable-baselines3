import experiment.gym_wrappers as gym_wrappers
import numpy as np
from tqdm import tqdm
import sys
import os
from abc import ABC, abstractmethod

if sys.version_info[0] == 2:
    import cPickle as pickle
else:
    import pickle
    import pathlib

from experiment.data_collectors import CollectorModule

class ExampleCollector(CollectorModule):
    def __init__(self, agents, num_dims, custom_ranges, num_divisions, num_agents, num_freq_saves, collect_seed, total_trials, save_disk, env_params):
        self.num_dims = num_dims
        self.num_divisions = num_divisions
        self.custom_ranges = custom_ranges
        self.num_agents = num_agents
        self.num_freq_saves = num_freq_saves
        self.collect_seed = collect_seed
        self.total_trials = total_trials
        self.save_disk = save_disk
        self.env_params = env_params
    
    def collect(self, agents):
        env = gym_wrappers.make("cartpole", "zero", "0", 
                                start_state_mode=self.env_params.start_state_mode,
                                start_states=self.env_params.start_states,
                                add_noise=self.env_params.add_noise,
                                seed=self.collect_seed)
        
        sh = [self.num_agents, 2] + [self.num_divisions] * self.num_dims

        self.divided_ranges = []
        for _ranges in self.custom_ranges:
            divided_range = np.array([])
            first = True
            for _min, _max, _num in _ranges:                
                linsp = np.linspace(_min, _max, _num)
                if not first:
                    linsp = linsp[1:]
                divided_range = np.concatenate((divided_range, linsp))
                first = False
            self.divided_ranges.append(divided_range)

        counts = np.zeros(sh, dtype=float)
        
        #count = 0
        for k in range(self.num_agents):
            print("agent # {}".format(k))
            for j in tqdm(range(self.total_trials)):
                # inner_counts = np.zeros(sh, dtype=int32)
                obs = env.reset()                
                env.manual_set(obs)
                for i in range(1000):
                    action, _states = agents[k][self.num_freq_saves-1].predict(obs, deterministic=True)
                    obs, reward, done, info = env.step(action)
                    env.render()
                    indexes = get_index(self.divided_ranges, obs)
                    counts[tuple([k, action] + indexes)] += 1
                    if done:
                        break
                # old_mean = mean
                # old_std_sq = std_sq
                # mean = old_mean + (inner_counts - old_mean) / (count+1)
                # if count >= 1:
                #     std_sq = (count - 1) * old_std_sq / count + (inner_counts - old_mean) ** 2 / count
                # count += 1
        if self.save_disk:
            with open('data.pickle', 'wb') as handle:
                pickle.dump({"ranges":self.divided_ranges, "counts":counts}, handle)
            print("saved file to {}".format(os.getcwd()))

        return self.divided_ranges, counts


