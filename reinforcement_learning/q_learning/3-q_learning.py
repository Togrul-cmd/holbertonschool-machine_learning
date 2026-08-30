#!/usr/bin/env python3
"""
Module containing the train function for Q-learning.
"""
import numpy as np
epsilon_greedy = __import__('2-epsilon_greedy').epsilon_greedy


def train(env, Q, episodes=5000, max_steps=100, alpha=0.1,
          gamma=0.99, epsilon=1, min_epsilon=0.1, epsilon_decay=0.05):
    """
    Performs Q-learning.

    Args:
        env: The FrozenLakeEnv instance.
        Q (numpy.ndarray): The Q-table.
        episodes (int): Total number of episodes to train over.
        max_steps (int): Maximum number of steps per episode.
        alpha (float): The learning rate.
        gamma (float): The discount rate.
        epsilon (float): Initial threshold for epsilon greedy.
        min_epsilon (float): Minimum value that epsilon should decay to.

    Returns:
        Q (numpy.ndarray): The updated Q-table.
        total_rewards (list): A list containing the rewards per episode.
    """
    total_rewards = []
    initial_epsilon = epsilon

    for episode in range(episodes):
        state, _ = env.reset()
        episode_reward = 0

        for step in range(max_steps):
            # Select action using epsilon-greedy strategy
            action = epsilon_greedy(Q, state, epsilon)

            # Take the action
            next_state, reward, terminated, truncated, _ = env.step(action)

            # When the agent falls in a hole, update the reward to -1
            if terminated and reward == 0.0:
                reward = -1

            # Update the Q-table using the Q-learning formula
            Q[state, action] = Q[state, action] + alpha * (
                reward + gamma * np.max(Q[next_state, :]) - Q[state, action]
            )

            episode_reward += reward
            state = next_state

            # End the episode if the agent reached a terminal state
            if terminated or truncated:
                break

        # Update epsilon with exponential decay between episodes
        epsilon = min_epsilon + (initial_epsilon - min_epsilon) * \
            np.exp(-epsilon_decay * episode)

        total_rewards.append(episode_reward)

    return Q, total_rewards
