import gym
import numpy as np
from gym.envs.registration import register
from .cart_pole import ModifiedCartPoleEnv, StartStateMode
from copy import deepcopy

ENTRY_POINT_MAP = {
    "cartpole": "experiment.gym_wrappers.cart_pole:ModifiedCartPoleEnv",
    "mountain_car": "experiment.gym_wrappers.mountain_car:TwoActionMountainCarEnv"
}

register_args = ["frame_skip", "episode_length", "max_episode_steps", "reward_threshold"]

entry_points_register_kwargs_defaults = {
    "cartpole": {
        "frame_skip": 1,
        "episode_length": 1000
    },
    "mountain_car": {
        "max_episode_steps": 200,
        "reward_threshold": -110.0
    }
}

common_kwargs_defaults = {       
}

entry_points_kwargs_defaults = {
    "cartpole": {
        "seed": 1,
        "add_noise": False,
        "start_state_mode": None,
        "start_states": None
    },
    "mountain_car": {        
    }
}


def make(
    domain_name,
    task_name,
    idx,
    **kwargs
):
    env_id = "gym_%s_%s_factor_%s-v1" % (domain_name, task_name, idx)

    register_kwargs = dict()

    for key in list(kwargs.keys()):
        if key in register_args:
            register_kwargs[key] = deepcopy(kwargs[key])
            del kwargs[key]

    env_kwargs = dict()
    env_kwargs.update(common_kwargs_defaults)
    assert domain_name in entry_points_kwargs_defaults
    env_kwargs.update(entry_points_kwargs_defaults[domain_name])
    env_kwargs.update(kwargs)
    
    assert domain_name in entry_points_register_kwargs_defaults
    register_kwargs.update(entry_points_register_kwargs_defaults[domain_name])
    if "max_episode_steps" not in register_kwargs:
        assert "episode_length" in register_kwargs
        assert "frame_skip" in register_kwargs
        # shorten episode length
        register_kwargs["max_episode_steps"] = (register_kwargs["episode_length"] + register_kwargs["frame_skip"] - 1) // register_kwargs["frame_skip"]
        del register_kwargs["episode_length"]
        del register_kwargs["frame_skip"]

    if env_id not in gym.envs.registry.env_specs:
        register(
            id=env_id,
            entry_point=ENTRY_POINT_MAP[domain_name],
            kwargs=env_kwargs,
            **register_kwargs
        )
    return gym.make(env_id)
