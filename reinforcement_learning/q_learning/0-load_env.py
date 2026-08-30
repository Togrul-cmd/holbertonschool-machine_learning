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
    # Updated to include render_mode="ansi" as per part 4 instructions
    env = gym.make(
        "FrozenLake-v1",
        desc=desc,
        map_name=map_name,
        is_slippery=is_slippery,
        render_mode="ansi"
    )

    return env
