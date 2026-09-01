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
            step_res = env.step(action)
            next_state = step_res[0]
            reward = step_res[1]
            done = step_res[2]
            if len(step_res) > 4:
                done = done or step_res[3]

            episode.append((state, reward))
            if done:
                break
            state = next_state

        G = 0
        states = [x[0] for x in episode]
        for t, (state, reward) in enumerate(reversed(episode)):
            G = gamma * G + reward
            idx = len(episode) - 1 - t
            if state not in states[:idx]:
                V[state] = V[state] + alpha * (G - V[state])

    return V
