#!/usr/bin/env python3
"""Defines the SARSA(lambda) algorithm for Q-value estimation"""
import numpy as np


def sarsa_lambtha(env, Q, lambtha, episodes=5000, max_steps=100,
                  alpha=0.1, gamma=0.99, epsilon=1, min_epsilon=0.1,
                  epsilon_decay=0.05):
    """
    Performs the SARSA(lambda) algorithm

    Args:
        env: the environment instance
        Q: numpy.ndarray of shape (s,a) containing the Q table
        lambtha: the eligibility trace factor
        episodes: total number of episodes to train over
        max_steps: maximum number of steps per episode
        alpha: the learning rate
        gamma: the discount rate
        epsilon: the initial threshold for epsilon greedy
        min_epsilon: the minimum value that epsilon should decay to
        epsilon_decay: the decay rate for updating epsilon between
            episodes

    Returns:
        Q, the updated Q table
    """
    initial_epsilon = epsilon
    n_actions = Q.shape[1]

    def epsilon_greedy(state, epsilon):
        if np.random.uniform(0, 1) < epsilon:
            return np.random.randint(0, n_actions)
        return np.argmax(Q[state])

    for ep in range(episodes):
        state, _ = env.reset()
        eligibility = np.zeros_like(Q)
        action = epsilon_greedy(state, epsilon)

        for step in range(max_steps):
            next_state, reward, terminated, truncated, _ = env.step(
                action)
            next_action = epsilon_greedy(next_state, epsilon)

            delta = (reward + gamma * Q[next_state, next_action]
                     - Q[state, action])
            eligibility[state, action] += 1

            Q = Q + alpha * delta * eligibility
            eligibility = gamma * lambtha * eligibility

            state = next_state
            action = next_action
            if terminated or truncated:
                break

        epsilon = (min_epsilon + (initial_epsilon - min_epsilon)
                   * np.exp(-epsilon_decay * ep))

    return Q
