#!/usr/bin/env python3
"""
Module containing the play function for a trained agent.
"""
import numpy as np


def play(env, Q, max_steps=100):
    """
    Has the trained agent play an episode.

    Args:
        env: The FrozenLakeEnv instance.
        Q (numpy.ndarray): The Q-table.
        max_steps (int): The maximum number of steps in the episode.

    Returns:
        total_rewards (float): The total rewards for the episode.
        rendered_outputs (list): A list of rendered outputs representing
                                 the board state at each step.
    """
    # Initialize the environment and get the starting state
    state, _ = env.reset()

    total_rewards = 0.0
    rendered_outputs = []

    # Append the initial state of the board
    rendered_outputs.append(env.render())

    for step in range(max_steps):
        # Always exploit the Q-table to find the best action
        action = np.argmax(Q[state, :])

        # Take the action
        next_state, reward, terminated, truncated, _ = env.step(action)

        # Append the new board state after taking the action
        rendered_outputs.append(env.render())

        total_rewards += reward
        state = next_state

        # End the episode if the agent reached a goal or fell in a hole
        if terminated or truncated:
            break

    return total_rewards, rendered_outputs
