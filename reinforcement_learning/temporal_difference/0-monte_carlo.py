#!/usr/bin/env python3
"""
Monte Carlo module
"""
import numpy as np


def monte_carlo(env, V, policy, episodes=5000, max_steps=100,
                alpha=0.1, gamma=0.99):
    """
    Performs the Monte Carlo algorithm for policy evaluation.

    Parameters:
    - env: environment instance
    - V: numpy.ndarray of shape (s,) containing the value estimate
    - policy: function that takes in a state and returns the next action
    - episodes: total number of episodes to train over
    - max_steps: maximum number of steps per episode
    - alpha: learning rate
    - gamma: discount rate

    Returns:
    - V: the updated value estimate
    """
    for ep in range(episodes):
        state = env.reset()
        if isinstance(state, tuple):
            state = state[0]

        episode = []
        for step in range(max_steps):
            action = policy(state)
            res = env.step(action)
            next_state = res[0]
            reward = res[1]
            done = res[2]

            # Compatibility for different Gym/Gymnasium versions
            if len(res) > 4:
                done = done or res[3]

            episode.append((state, reward))
            if done:
                break
            state = next_state

        G = 0
        # Every-Visit Monte Carlo prediction calculation
        for t in range(len(episode) - 1, -1, -1):
            s_t, r_t = episode[t]
            G = gamma * G + r_t
            # Update applied on every visit (First-Visit restriction removed)
            V[s_t] = V[s_t] + alpha * (G - V[s_t])

    return V
