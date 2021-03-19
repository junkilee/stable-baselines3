import os
import hydra
import logging
import copy
from abc import ABC, abstractmethod
from tqdm import tqdm
import sys

if sys.version_info[0] == 2:
    import cPickle as pickle
else:
    import pickle
    import pathlib

from torch import int32

from stable_baselines3 import A2C, DDPG, DQN, HER, PPO, SAC, TD3
import experiment.gym_wrappers as gym_wrappers

import numpy as np
import matplotlib.pyplot as plt

ALGOS = {
    "a2c": A2C,
    "ddpg": DDPG,
    "dqn": DQN,
    "ppo": PPO,
    "her": HER,
    "sac": SAC,
    "td3": TD3,
}


def load_data(filename):
    with open(filename, 'rb') as handle:
            data = pickle.load(handle)
    return data    


class ExampleCollector(object):
    def __init__(self, cfg):
        self.agents = []
        self._setup_agent(cfg)
        self.env_params = cfg.example_gen.env_params
        self.num_agents = cfg.example_gen.num_agents
        self.num_freq_saves = cfg.example_gen.num_freq_saves
        self.collect_seed = cfg.example_gen.collect_seed
        self.total_trials = cfg.example_gen.total_trials
        self.save_disk = cfg.example_gen.save_disk
        self.example_save_freqs = cfg.example_gen.example_save_freqs
        self.render = cfg.example_gen.render

    def _setup_agent(self, cfg):        
        # setup training environment if necessary        
        self.train_env = gym_wrappers.make(
            cfg.env, 
            cfg.agents.task_name, 
            "0",  
            start_state_mode=eval(cfg.agents.env_params.start_state_mode),
            start_states=cfg.agents.env_params.start_states,
            seed=cfg.agents.seed)
        # setup agents
        self.num_agents = cfg.agents.num_agents
        for i in range(self.num_agents):
            self.agents += [ALGOS[cfg.agents.algorithm](
                cfg.agents.policy,
                self.train_env,
                verbose=1,
                seed=cfg.agents.seed + i * 7,
            )]
        # train or load agents
        if not cfg.agents.load_agent:
            for i in range(self.num_agents):
                if cfg.agents.save_agent:
                    self.agents[i].learn(total_timesteps=cfg.agents.timesteps,
                                        save_freq=cfg.agents.save_freq,
                                        save_path=os.path.join(cfg.agents.save_dir, str(i)))
                else:
                    self.agents[i].learn(total_timesteps=cfg.agents.timesteps)
        else:
            if cfg.agents.save_freq != -1:
                num_saves = cfg.agents.timesteps // cfg.agents.save_freq
                for i in range(self.num_agents):
                    seed_agent = self.agents[i]
                    self.agents[i] = list()
                    for j in range(num_saves):
                        idx_j = cfg.agents.save_freq // 1000 * (j + 1)
                        copied_agent = copy.deepcopy(seed_agent)
                        self.agents[i].append(copied_agent.load(os.path.join(cfg.agents.save_dir, str(i)) + "_"+str(idx_j)))
            else:
                for i in range(self.num_agents):
                    self.agents[i].load(os.path.join(cfg.agents.save_dir, str(i)))


    def collect(self):
        env = gym_wrappers.make("cartpole", "zero", "0",
                                start_state_mode=self.env_params.start_state_mode,
                                start_states=self.env_params.start_states,
                                add_noise=self.env_params.add_noise,
                                seed=self.collect_seed)
        data = []
        
        for j in tqdm(range(self.total_trials)):
            # inner_counts = np.zeros(sh, dtype=int32)
            obs = env.reset()
            env.manual_set(obs)
            for i in range(1000):
                action, _states = self.agents[0][self.num_freq_saves-1].predict(obs, deterministic=True)
                data.append((obs, action))
                obs, reward, done, info = env.step(action)
                if self.render:
                    env.render()
                if done:
                    break
        if self.save_disk:
            with open('example_data_{}.pickle'.format(j), 'wb') as handle:
                pickle.dump(data, handle)
            print("saved file to {}".format(os.getcwd()))
        return

simple_example = [ # L, (right) C
    ([1, 0, 1, 0], 0),
    ([1, 0, 0, 1], 0),
    ([0, 0, 1, 0], 1),
    ([1, 1, 0, 0], 0),
    ([0, 0, 0, 1], 1),
    ([1, 1, 1, 1], 0),
    ([0, 1, 1, 0], 0),
    ([0, 0, 1, 1], 1)
]

@hydra.main(config_path="config", config_name="example_generator")
def main(cfg):
    logging.info("Working directory : {}".format(os.getcwd()))
    example_collector = ExampleCollector(cfg)
    example_collector.collect()

if __name__ == "__main__":
    main()