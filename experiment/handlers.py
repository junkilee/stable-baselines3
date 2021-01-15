import os
import hydra
import logging
import copy
from abc import ABC, abstractmethod
from tqdm import tqdm

from torch import int32

from stable_baselines3 import A2C, DDPG, DQN, HER, PPO, SAC, TD3
import experiment.gym_wrappers as gym_wrappers

import numpy as np
import matplotlib.pyplot as plt
import sys

if sys.version_info[0] == 2:
    import cPickle as pickle
else:
    import pickle
    import pathlib


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
        if cfg.collector_module != "None":
            self.collector_module = hydra.utils.instantiate(cfg.collector_module)
        else:
            self.collector_module = None
        if cfg.test_module != "None":
            self.test_module = hydra.utils.instantiate(cfg.test_module)
        else:
            self.test_module = None

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
        return ranges, data


class CollectorModule(ABC):
    @abstractmethod
    def collect(self, agents):
        ranges = []
        data = []
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


def get_index(div_ranges, obs):
    result = []
    for i, _range in enumerate(div_ranges):
        idx = -1
        for j, x in enumerate(_range):
            if obs[i] <= x:
                idx = j
                break
        if idx == -1:
            result.append(len(_range) - 1)
        else:
            result.append(idx)
    return result


class DataCollector(CollectorModule):
    def __init__(self, num_dims, ranges, num_divides, num_agents, num_freq_saves, collect_seed, total_trials, env_params):
        self.num_dims = num_dims
        self.num_divides = num_divides
        self.ranges = ranges
        self.num_agents = num_agents
        self.num_freq_saves = num_freq_saves
        self.collect_seed = collect_seed
        self.total_trials = total_trials
        self.env_params = env_params
    
    def collect(self, agents):
        env = gym_wrappers.make("cartpole", "zero", "0", 
                                start_state_mode=self.env_params.start_state_mode,
                                start_states=self.env_params.start_states, 
                                seed=self.collect_seed)
        
        sh = [self.num_agents, 2] + [self.num_divides] * self.num_dims

        divided_ranges = []
        for i in range(self.num_dims):
            divided_ranges.append(np.linspace(self.ranges[i][0], self.ranges[i][1], self.num_divides))
        
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
                    indexes = get_index(divided_ranges, obs)
                    counts[tuple([k, action] + indexes)] += 1
                    if done:
                        break
                # old_mean = mean
                # old_std_sq = std_sq
                # mean = old_mean + (inner_counts - old_mean) / (count+1)
                # if count >= 1:
                #     std_sq = (count - 1) * old_std_sq / count + (inner_counts - old_mean) ** 2 / count
                # count += 1

        return divided_ranges, counts


class CustomRangeDataCollector(CollectorModule):
    def __init__(self, num_dims, custom_ranges, num_divisions, num_agents, num_freq_saves, collect_seed, total_trials, save_disk, env_params):
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
                    #env.render()
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

class ExplanationGenerator(object):
    def __init__(self, data):
        self.data = data

    def explain(self, state=None):
        avg = []
        stderr = []

        ranges = []
        ranges += [np.linspace(-2.4, 2.4, 20)]
        ranges += [np.linspace(-4.8, 4.8, 20)]
        ranges += [np.linspace(-0.209, 0.209, 20)]
        ranges += [np.linspace(-0.418, 0.418, 20)]

        dp = [10, 10, 10, 10]
        if state is None:
            pd = np.sum(np.sum(self.data, axis=0), axis=0).flatten()
            pd = pd / np.sum(pd)
            pick = int(np.nonzero(np.random.multinomial(1, pd))[0])

            dp = []

            for i in range(4):
                dp = [pick % 20] + dp
                pick = pick // 20
            #dp = [10, 9, 10, 11]
        else:
            pass
            
            
        print(dp)
        real_dp = []
        probs = []
        errs = []
        to_probs = []
        to_errs = []

        for i in range(4):
            real_dp += [(ranges[i][dp[i]-1] + ranges[i][dp[i]])/2]

        print(real_dp)

        for action in range(2):
            probs += [np.average(np.sum(self.data[:, action, dp[0], :, :, :], axis=(1,2,3)) / np.sum(self.data[:, action, :, :, :, :], axis=(1,2,3,4)))]
            errs += [np.std(np.sum(self.data[:, action, dp[0], :, :, :], axis=(1,2,3)) / np.sum(self.data[:, action, :, :, :, :], axis=(1,2,3,4)))]
            probs += [np.average(np.sum(self.data[:, action, :, dp[1], :, :], axis=(1,2,3)) / np.sum(self.data[:, action, :, :, :, :], axis=(1,2,3,4)))]
            errs += [np.std(np.sum(self.data[:, action, :, dp[1], :, :], axis=(1,2,3)) / np.sum(self.data[:, action, :, :, :, :], axis=(1,2,3,4)))]
            probs += [np.average(np.sum(self.data[:, action, :, :, dp[2], :], axis=(1,2,3)) / np.sum(self.data[:, action, :, :, :, :], axis=(1,2,3,4)))]
            errs += [np.std(np.sum(self.data[:, action, :, :, dp[2], :], axis=(1,2,3)) / np.sum(self.data[:, action, :, :, :, :], axis=(1,2,3,4)))]
            probs += [np.average(np.sum(self.data[:, action, :, :, :, dp[3]], axis=(1,2,3)) / np.sum(self.data[:, action, :, :, :, :], axis=(1,2,3,4)))]
            errs += [np.std(np.sum(self.data[:, action, :, :, :, dp[3]], axis=(1,2,3)) / np.sum(self.data[:, action, :, :, :, :], axis=(1,2,3,4)))]
            
            avg += [np.average(self.data[:, action, :, dp[1], dp[2], dp[3]], axis=0)]
            stderr += [np.std(self.data[:, 0, :, dp[1], dp[2], dp[3]], axis=0)]
            avg += [np.average(self.data[:, action, dp[0], :, dp[2], dp[3]], axis=0)]
            stderr += [np.std(self.data[:, 0, dp[0], :, dp[2], dp[3]], axis=0)]
            avg += [np.average(self.data[:, action, dp[0], dp[1], :, dp[3]], axis=0)]
            stderr += [np.std(self.data[:, 0, dp[0], dp[1], :, dp[3]], axis=0)]
            avg += [np.average(self.data[:, action, dp[0], dp[1], dp[2], :], axis=0)]
            stderr += [np.std(self.data[:, 0, dp[0], dp[1], dp[2], :], axis=0)]
        to_probs += [np.average(np.sum(self.data[:, :, dp[0], :, :, :], axis=(1,2,3,4)) / np.sum(self.data[:, :, :, :, :, :], axis=(1,2,3,4,5)))]
        to_errs += [np.std(np.sum(self.data[:, :, dp[0], :, :, :], axis=(1,2,3,4)) / np.sum(self.data[:, :, :, :, :, :], axis=(1,2,3,4,5)))]
        to_probs += [np.average(np.sum(self.data[:, :, :, dp[1], :, :], axis=(1,2,3,4)) / np.sum(self.data[:, :, :, :, :, :], axis=(1,2,3,4,5)))]
        to_errs += [np.std(np.sum(self.data[:, :, :, dp[1], :, :], axis=(1,2,3,4)) / np.sum(self.data[:, :, :, :, :, :], axis=(1,2,3,4,5)))]
        to_probs += [np.average(np.sum(self.data[:, :, :, :, dp[2], :], axis=(1,2,3,4)) / np.sum(self.data[:, :, :, :, :, :], axis=(1,2,3,4,5)))]
        to_errs += [np.std(np.sum(self.data[:, :, :, :, dp[2], :], axis=(1,2,3,4)) / np.sum(self.data[:, :, :, :, :, :], axis=(1,2,3,4,5)))]
        to_probs += [np.average(np.sum(self.data[:, :, :, :, :, dp[3]], axis=(1,2,3,4)) / np.sum(self.data[:, :, :, :, :, :], axis=(1,2,3,4,5)))]
        to_errs += [np.std(np.sum(self.data[:, :, :, :, :, dp[3]], axis=(1,2,3,4)) / np.sum(self.data[:, :, :, :, :, :], axis=(1,2,3,4,5)))]

        probs = np.array(probs)
        errs = np.array(errs)

        to_probs = np.array(to_probs)
        to_errs = np.array(to_errs)

        
        import matplotlib.pyplot as plt
        import matplotlib.patches as patches

        rad_90 = 3.14159265/2
        fig,ax = plt.subplots(1)
        rect = patches.Rectangle((real_dp[0]-0.15,-0.12),0.3,0.24,linewidth=1.5,edgecolor='black',facecolor='white')
        ax.add_patch(rect)
        ax.plot([real_dp[0], real_dp[0] + np.cos(real_dp[2] + rad_90)], [0, np.sin(real_dp[2] + rad_90)], color='black', linewidth=2.5)
        ax.axhline(y=0)
        ax.set_ylim([-0.8, 2.4])
        ax.set_xlim([-2.4, 2.4])
        plt.savefig("cart.png")
        plt.show()

        names = ['cart lateral pos', 'cart lateral vel', 'pole angle', 'pole angular vel']

        print(probs[0:4])
        plt.bar(names, probs[0:4], yerr=errs[0:4])
        plt.savefig("prob_a0_res.png")
        plt.show()

        print(probs[4:8])
        plt.bar(names, probs[4:8], yerr=errs[4:8])
        plt.savefig("prob_a1_res.png")
        plt.show()

        print(to_probs)
        plt.bar(names, to_probs, yerr=to_errs)
        plt.savefig("prob_to_res.png")
        plt.show()

        saves = []
        for i in range(4):
            output_str = ""
            for j in range(4):
                if i != j:
                    output_str += " x[{}]={:.4f},".format(j, real_dp[j])            
            output_str = output_str[:-1]
            plt.title(names[i] + "\n" + "p(x[{}]| action = right,{})".format(i, output_str))
            diff = (ranges[i][1] - ranges[i][0]) / 2
            
            yesyes = False
            if np.sum(avg[i]) == 0:
                plt.plot(ranges[i] - diff, avg[i])
            else:
                yesyes = True
                plt.plot(ranges[i] - diff, avg[i] / np.sum(avg[i]))
                #plt.fill_between(ranges[i] - diff, avg[i]/ np.sum(avg[i])-stderr[i]/ np.sum(avg[i]), avg[i]/ np.sum(avg[i])+stderr[i]/ np.sum(avg[i]), fc='b')
            if np.sum(avg[i+4]) == 0:
                plt.plot(ranges[i] - diff, avg[i+4])
                yesyes = False
            else:
                plt.plot(ranges[i] - diff, avg[i+4]/np.sum(avg[i+4]), c='r')
                yesyes = True
                #plt.fill_between(ranges[i] - diff, avg[i+4]/np.sum(avg[i+4])-stderr[i+4]/np.sum(avg[i+4]), avg[i+4]/np.sum(avg[i+4])+stderr[i+4]/np.sum(avg[i+4]), fc='r')
            
            #Testing with Point Biserial Correlation
            r_pb = 0
            if yesyes:
                ss0 = []
                ss1 = []
                for idx, val in enumerate(avg[i]):
                    if idx is not 0:
                        ss0 += [(ranges[i][idx-1] + ranges[i][idx])/2] * int(val)            
                        ss1 += [(ranges[i][idx-1] + ranges[i][idx])/2] * int(avg[i+4][idx])
                m0 = np.average(ss0)
                m1 = np.average(ss1)
                sy = np.std(ss0+ss1)
                n0 = np.sum(avg[i])
                n1 = np.sum(avg[i+4])
                nn = n0 + n1
                r_pb = (m0-m1)/sy * np.sqrt(n0/nn * n1/nn)
            
            
            print(dp[i])
            print("difference", abs(avg[i][dp[i]]/np.sum(avg[i]) - avg[i+4][dp[i]]/np.sum(avg[i+4])))
            plt.axvline(x=[ranges[i][dp[i]]-diff], c='g')
            plt.ylim([-0.05, 1.1])
            plt.savefig("result_{}.png".format(i))
            plt.show()
            if yesyes:        
                print("point biserial coefficient = {}".format(r_pb))
                if r_pb > 0.5:
                    saves += [(i, r_pb)]


@hydra.main(config_path="config", config_name="load", strict=True)
def main(cfg):
    logging.info("Working directory : {}".format(os.getcwd()))
    experiment = ExperimentHandler(cfg)
    experiment.run()

if __name__ == "__main__":
    main()