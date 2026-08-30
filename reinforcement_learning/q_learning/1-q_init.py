#!/usr/bin/env python3
"""
Module containing the q_init function.
"""
import numpy as np


def q_init(env):
    """
    Initializes the Q-table for a given environment.

    Args:
        env: The FrozenLakeEnv instance.

    Returns:
        The Q-table as a numpy.ndarray of zeros.
    """
    # Get the number of states and actions from the environment
    num_states = env.observation_space.n
    num_actions = env.action_space.n

    # Initialize a 2D numpy array of zeros with shape (states, actions)
    q_table = np.zeros((num_states, num_actions))

    return q_table
