#!/usr/bin/env python3
"""
Module containing the epsilon_greedy function.
"""
import numpy as np


def epsilon_greedy(Q, state, epsilon):
    """
    Uses epsilon-greedy to determine the next action.

    Args:
        Q (numpy.ndarray): The Q-table.
        state (int): The current state.
        epsilon (float): The epsilon to use for the calculation.

    Returns:
        int: The next action index.
    """
    # Sample a random float between 0 and 1
    p = np.random.uniform(0, 1)

    # If p is less than epsilon, explore
    if p < epsilon:
        # Pick a random action from all possible actions
        num_actions = Q.shape[1]
        action = np.random.randint(0, num_actions)
    # Otherwise, exploit
    else:
        # Pick the action with the highest Q-value for the current state
        action = np.argmax(Q[state, :])

    return int(action)
