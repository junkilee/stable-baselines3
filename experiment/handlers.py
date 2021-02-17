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


class ExperimentHandler(object):
    def __init__(self, cfg):        
        self.experiment_name = cfg.experiment_name
        self.cfg = cfg
        self.agents = []        
        if hasattr(cfg, 'collector_module') and cfg.collector_module != "None":
            self.collector_module = hydra.utils.instantiate(cfg.collector_module)
        else:
            self.collector_module = None
        if hasattr(cfg, 'test_module') and cfg.test_module != "None":
            self.test_module = hydra.utils.instantiate(cfg.test_module)
        else:
            self.test_module = None
        if hasattr(cfg, 'animator_module') and cfg.animator_module != "None":
            self.animator_module = hydra.utils.instantiate(cfg.animator_module)
        else:
            self.animator_module = None        

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

    def run(self):
        ranges, data = None, None        
        if self.agents == []:
            self._setup_agent(self.cfg)
        if self.test_module is not None:
            self.test_module.run(self.agents)
        if self.collector_module is not None:
            ranges, data = self.collector_module.collect(self.agents)
        if self.animator_module is not None:
            self.animator_module.collect(self.agents)
        
        return ranges, data



class TestModule(ABC):
    @abstractmethod
    def run(self, agents):
        pass


class PriorTestModule(TestModule):
    def __init__(self, num_agents, num_freq_saves, total_train_timesteps, save_freq, test_seed):
        self.num_agents = num_agents
        self.num_freq_saves = num_freq_saves
        self.total_train_timesteps = total_train_timesteps
        self.save_freq = save_freq
        self.test_seed = test_seed

    def run(self, agents):
        env = gym_wrappers.make("cartpole", "uniform", "0", 
                                start_state_mode =gym_wrappers.StartStateMode.UNIFORM,
                                start_states=[-2.4,2.4], seed=self.test_seed)

        num_x_datapoints = 20
        num_y_datapoints = 10

        data = np.zeros((num_x_datapoints, 2))
        data2 = np.zeros((num_x_datapoints))
        
        idx = 0
        xrange = np.linspace(-0.418, 0.418, num_x_datapoints)        
        for x in xrange:
            for k in range(self.num_agents):
                for j in range(num_y_datapoints):    
                    obs = env.reset()
                    #obs[0] = x + np.random.uniform(low=-0.05, high=0.05)
                    env.manual_set(obs)
                    for i in range(1000):
                        action, _states = agents[k][self.num_freq_saves-1].predict(obs, deterministic=True)
                        obs, reward, done, info = env.step(action)                        
                        x = obs[2]
                        for idx, p in enumerate(xrange):
                            if x < p:
                                data[idx, int(action)] += 1
                        
                        if done:        
                            break
            idx += 1

        
        env.close()
        real_data = data[:,0] / (data[:,0] + data[:,1])
        sum_data = (data[:,0] + data[:,1])
        #plt.plot(xrange, real_data)
        #plt.savefig("results1.png")        

        plt.plot(xrange, sum_data / np.sum(sum_data))
        plt.savefig("results2.png")        
        plt.close()
        pass


@hydra.main(config_path="config", config_name="load", strict=True)
def main(cfg):
    logging.info("Working directory : {}".format(os.getcwd()))
    experiment = ExperimentHandler(cfg)
    experiment.run()

if __name__ == "__main__":
    main()