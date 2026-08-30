#!/usr/bin/env python3
"""
Module containing the load_frozen_lake function.
"""
import gymnasium as gym


def load_frozen_lake(desc=None, map_name=None, is_slippery=False):
    """
    Loads the pre-made FrozenLakeEnv environment from gymnasium.

    Args:
        desc (list): A list of lists containing a custom map description.
        map_name (str): A string containing the pre-made map to load.
        is_slippery (bool): Determines if the ice is slippery.

    Returns:
        The gymnasium environment.
    """
    # If both desc and map_name are None, generate a random 8x8 map
    # using the function built into the gym namespace
    if desc is None and map_name is None:
        desc = gym.envs.toy_text.frozen_lake.generate_random_map(size=8)

    # Initialize and return the environment
    env = gym.make(
        "FrozenLake-v1",
        desc=desc,
        map_name=map_name,
        is_slippery=is_slippery
    )

    return env
